#![cfg(feature = "cuda")]

#[path = "support/combined_gemm_acceptance.rs"]
mod combined_gemm_acceptance;
#[path = "../../tests/common/gpu_quiet.rs"]
mod gpu_quiet;
#[path = "support/production_auto_cohort.rs"]
mod production_auto_cohort;

use std::cell::{Cell, RefCell};
use std::ffi::CStr;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaGraph, PushKernelArg};
use gpu_quiet::QuietGpu;
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_inference::{
    InferenceFwdOperands, InferenceShape, InferenceSm120HalfTile, InferenceTile, inference_forward,
    inference_forward_f32_legacy_baseline, inference_forward_with_tile,
};
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    PhysicalQualificationRequest, PhysicalQualificationRoute, Sm120Bk, Sm120ForcedRoute,
    Sm120LaunchOperands, Sm120MapRequest, Sm120Op, Sm120PhysicalRoute, Sm120Schedule, Sm120Shape,
    Sm120Stages, Sm120Tile, TcTile, Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages,
    Tf32PortableTile, Tf32Sm120Route, Tf32Sm120Stages, Tf32Sm120Tile, launch_sm120_tma_prepared,
    prepare_sm120_tensor_maps, prepare_sm120_tma_forced, presize_physical_qualification_suite,
    qualify_physical_launch, resolve_sm120_forced, tf32_route_specs,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    ModuleKind, PhysicalGemmBackend, PolicyDtype, ResolvedGemmOp, ResolvedGemmRoute,
    TUNING_TABLE_REVISION, digest_hex,
};
use production_auto_cohort::{
    ProductionAutoInventory, render_cohort_fragment, state_capacity_from_env,
};
use sha2::{Digest, Sha256};

const WARMUPS: usize = 10;
const ITERS: usize = 200;

// The Ada S3 pair below has its own mirrored-bracket protocol; the older
// harnesses above and below keep their original record meanings.
mod ada_s3_pair {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(super) enum Stage {
        PrePromotion42,
        PostAuto43,
    }

    impl Stage {
        fn schema(self) -> &'static str {
            match self {
                Self::PrePromotion42 => "MambaBiFixedAdaS3PairedV1",
                Self::PostAuto43 => "MambaBiFixedAdaS3PostAutoPairedV1",
            }
        }

        fn revision(self) -> u16 {
            match self {
                Self::PrePromotion42 => 42,
                Self::PostAuto43 => 43,
            }
        }

        fn controls(self) -> [&'static str; 3] {
            match self {
                Self::PrePromotion42 => [
                    "MAMBA_FIXED_ADA_S3_PAIR",
                    "MAMBA_FIXED_ADA_S3_WINDOWS",
                    "MAMBA_FIXED_ADA_S3_DTYPES",
                ],
                Self::PostAuto43 => [
                    "MAMBA_FIXED_ADA_S3_POST_PAIR",
                    "MAMBA_FIXED_ADA_S3_POST_WINDOWS",
                    "MAMBA_FIXED_ADA_S3_POST_DTYPES",
                ],
            }
        }

        fn arms(self) -> [&'static str; 3] {
            match self {
                Self::PrePromotion42 => ["AUTO", "S3", "Fast"],
                Self::PostAuto43 => ["Swizzle", "AUTO", "Fast"],
            }
        }

        fn ratios(self) -> [&'static str; 3] {
            match self {
                Self::PrePromotion42 => ["S3/AUTO", "AUTO/Fast", "S3/Fast"],
                Self::PostAuto43 => ["AUTO/Swizzle", "Swizzle/Fast", "AUTO/Fast"],
            }
        }
    }

    fn schedule(window: usize, start: usize) -> Vec<(usize, usize, usize, usize)> {
        assert!(start < 2);
        let reverse = (window + start) % 2 == 1;
        let comparisons = if reverse { [2, 1, 0] } else { [0, 1, 2] };
        let mut observations = Vec::with_capacity(12);
        for (traversal, comparison) in comparisons.into_iter().enumerate() {
            let (a, b) = [(0, 1), (2, 0), (2, 1)][comparison];
            let arms = if reverse { [b, a, a, b] } else { [a, b, b, a] };
            for (position, arm) in arms.into_iter().enumerate() {
                observations.push((traversal, comparison, position, arm));
            }
        }
        observations
    }

    const GUARD: usize = 256;

    fn control_allowed(stage: Stage, key: &str) -> bool {
        if stage.controls().contains(&key) {
            return true;
        }
        !key.starts_with("MAMBA_FIXED_ADA_")
            && !key.starts_with("MAMBA_FIXED_VENDOR_")
            && !matches!(
                key,
                "MAMBA_FIXED_AUTO_VENDOR_ROW"
                    | "MAMBA_FIXED_AUTO_VENDOR_CELL"
                    | "MAMBA_FIXED_AUTO_VENDOR_BIAS"
                    | "MAMBA_FIXED_HALF_TILE_CANDIDATE"
                    | "NVIDIA_TF32_OVERRIDE"
            )
    }

    fn validate_controls<'a>(
        stage: Stage,
        controls: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), String> {
        for key in controls {
            if !control_allowed(stage, key) {
                return Err(format!(
                    "stale control {key} forbidden in literal S3 experiment"
                ));
            }
        }
        Ok(())
    }

    fn config(stage: Stage, windows: &str, dtypes: &str) -> Result<(usize, Vec<usize>), String> {
        let windows = match windows {
            "1" => 1,
            "21" => 21,
            "101" => 101,
            _ => return Err("windows must be exactly 1, 21 or 101".into()),
        };
        let dtypes = fixed_ada_direct_pair_filter("S3 dtype", &["bf16", "f16"], Some(dtypes))?;
        match stage {
            Stage::PrePromotion42 if windows != 101 && dtypes != [0, 1] => {
                return Err("historical smoke/screen requires explicit bf16,f16".into());
            }
            Stage::PostAuto43 if !matches!(windows, 1 | 101) || dtypes != [0, 1] => {
                return Err("post-AUTO requires windows1/101 and explicit bf16,f16".into());
            }
            _ => {}
        }
        Ok((windows, dtypes))
    }

    #[test]
    fn pre42_and_post43_configuration_cannot_mute_screen_or_accept_implicit_windows() {
        assert_eq!(
            config(Stage::PrePromotion42, "21", "bf16,f16").unwrap(),
            (21, vec![0, 1])
        );
        assert_eq!(
            config(Stage::PrePromotion42, "101", "f16").unwrap(),
            (101, vec![1])
        );
        assert_eq!(
            config(Stage::PostAuto43, "1", "bf16,f16").unwrap(),
            (1, vec![0, 1])
        );
        assert_eq!(
            config(Stage::PostAuto43, "101", "bf16,f16").unwrap(),
            (101, vec![0, 1])
        );
        assert!(config(Stage::PostAuto43, "21", "bf16,f16").is_err());
        assert!(config(Stage::PostAuto43, "101", "bf16").is_err());
        assert!(config(Stage::PostAuto43, "101", "f16").is_err());
        for w in ["", "0", "20", "021", " 21", "100", "102"] {
            assert!(config(Stage::PrePromotion42, w, "bf16,f16").is_err());
        }
        for d in ["", "f32", "bf16,bf16", "bf16,", "f16"] {
            assert!(config(Stage::PrePromotion42, "21", d).is_err());
        }
    }

    #[test]
    fn pre42_and_post43_stage_controls_and_directions_are_disjoint() {
        let pre = Stage::PrePromotion42;
        let post = Stage::PostAuto43;
        assert_eq!(pre.schema(), "MambaBiFixedAdaS3PairedV1");
        assert_eq!(post.schema(), "MambaBiFixedAdaS3PostAutoPairedV1");
        assert_eq!(pre.revision(), 42);
        assert_eq!(post.revision(), 43);
        assert_eq!(pre.arms(), ["AUTO", "S3", "Fast"]);
        assert_eq!(post.arms(), ["Swizzle", "AUTO", "Fast"]);
        assert_eq!(pre.ratios(), ["S3/AUTO", "AUTO/Fast", "S3/Fast"]);
        assert_eq!(post.ratios(), ["AUTO/Swizzle", "Swizzle/Fast", "AUTO/Fast"]);
        assert!(validate_controls(pre, pre.controls()).is_ok());
        assert!(validate_controls(post, post.controls()).is_ok());
        assert!(validate_controls(pre, post.controls()).is_err());
        assert!(validate_controls(post, pre.controls()).is_err());
        for stale in [
            "MAMBA_FIXED_AUTO_VENDOR_ROW",
            "MAMBA_FIXED_AUTO_VENDOR_CELL",
            "MAMBA_FIXED_AUTO_VENDOR_BIAS",
            "MAMBA_FIXED_HALF_TILE_CANDIDATE",
            "MAMBA_FIXED_VENDOR_LEGACY_CONTROL",
            "NVIDIA_TF32_OVERRIDE",
        ] {
            assert!(
                validate_controls(pre, [stale]).is_err(),
                "pre accepted {stale}"
            );
            assert!(
                validate_controls(post, [stale]).is_err(),
                "post accepted {stale}"
            );
        }
    }

    fn unchanged_inputs(
        saved_a: &[u8],
        saved_b: &[u8],
        observed_a: &[u8],
        observed_b: &[u8],
    ) -> Result<(), String> {
        if observed_a != saved_a {
            return Err("immutable A changed before timing".into());
        }
        if observed_b != saved_b {
            return Err("immutable B changed before timing".into());
        }
        Ok(())
    }

    #[test]
    fn pre_timing_input_gate_rejects_changed_a_or_b_bytes() {
        let saved_a = [1, 2, 3, 4];
        let saved_b = [5, 6, 7, 8];
        assert!(unchanged_inputs(&saved_a, &saved_b, &saved_a, &saved_b).is_ok());
        assert!(unchanged_inputs(&saved_a, &saved_b, &[1, 2, 0, 4], &saved_b).is_err());
        assert!(unchanged_inputs(&saved_a, &saved_b, &saved_a, &[5, 6, 0, 8]).is_err());
    }

    fn pair_ratio(samples: &[(usize, f64)], comparison: usize) -> Result<f64, String> {
        let (a, b) = [(0, 1), (2, 0), (2, 1)][comparison];
        if samples.len() != 4
            || samples.iter().any(|(_, t)| !t.is_finite() || *t <= 0.0)
            || samples.iter().filter(|(arm, _)| *arm == a).count() != 2
            || samples.iter().filter(|(arm, _)| *arm == b).count() != 2
        {
            return Err("invalid mirrored observations".into());
        }
        Ok(samples
            .iter()
            .filter(|(arm, _)| *arm == b)
            .map(|(_, t)| t)
            .sum::<f64>()
            / samples
                .iter()
                .filter(|(arm, _)| *arm == a)
                .map(|(_, t)| t)
                .sum::<f64>())
    }

    #[test]
    fn ratios_use_sum_of_two_observations_and_reject_invalid_times() {
        assert_eq!(
            pair_ratio(&[(0, 10.0), (1, 4.0), (1, 8.0), (0, 20.0)], 0).unwrap(),
            0.4
        );
        for t in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(pair_ratio(&[(0, 10.0), (1, t), (1, 8.0), (0, 20.0)], 0).is_err());
        }
        assert!(pair_ratio(&[(0, 10.0), (2, 4.0), (1, 8.0), (0, 20.0)], 0).is_err());
        let mut loss = vec![0.8; 21];
        loss[19] = 1.2;
        loss[20] = 1.4;
        assert_eq!(
            (percentile(&loss, 0.5), percentile(&loss, 0.95)),
            (0.8, 1.2)
        );
    }

    fn upload(ctx: &GpuCtx, ptr: u64, bytes: &[u8]) {
        assert_eq!(
            unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    ptr,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    ctx.stream.cu_stream(),
                )
            },
            cudarc::driver::sys::CUresult::CUDA_SUCCESS
        );
        ctx.stream.synchronize().expect("S3 upload sync");
    }

    fn raw(ctx: &GpuCtx, owner: &DtypedBuf) -> Vec<u8> {
        let all = fixed_explicit_vendor_raw_bytes(ctx, owner);
        assert!(
            all[..GUARD]
                .iter()
                .chain(all[all.len() - GUARD..].iter())
                .all(|b| *b == 0x5a),
            "S3 output allocation guard overwritten"
        );
        all[GUARD..all.len() - GUARD].to_vec()
    }

    fn poison(ctx: &GpuCtx, owner: &DtypedBuf, expected: &[u8]) {
        let complement: Vec<_> = expected.iter().map(|b| !b).collect();
        upload(ctx, owner.cached_ptr() + GUARD as u64, &complement);
        let observed = raw(ctx, owner);
        assert_eq!(observed, complement, "S3 poison upload readback");
        assert!(
            observed
                .as_chunks::<2>()
                .0
                .iter()
                .zip(expected.as_chunks::<2>().0)
                .all(|(a, b)| a != b),
            "every output storage word must differ before independent replay"
        );
    }

    fn output_gate(
        ctx: &GpuCtx,
        owner: &DtypedBuf,
        expected: &[u8],
        replay: impl FnOnce(),
    ) -> bool {
        poison(ctx, owner, expected);
        replay();
        raw(ctx, owner) == expected
    }

    fn half_bits(raw: &[u8], dtype: WeightDtype) -> Vec<u32> {
        raw.as_chunks::<2>()
            .0
            .iter()
            .map(|v| {
                let bits = u16::from_le_bytes(*v);
                match dtype {
                    WeightDtype::Bf16 => half::bf16::from_bits(bits).to_f32().to_bits(),
                    WeightDtype::F16 => half::f16::from_bits(bits).to_f32().to_bits(),
                    _ => unreachable!(),
                }
            })
            .collect()
    }

    fn fast(ctx: &GpuCtx, ops: InferenceFwdOperands, shape: InferenceShape) {
        use cudarc::cublas::{result, sys};
        assert!(ops.bias_ptr.is_none());
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            result::gemm_ex(
                *ctx.blas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_N,
                shape.n as i32,
                shape.m as i32,
                shape.k as i32,
                (&alpha as *const f32).cast(),
                ops.w.ptr as *const _,
                ops.w.dtype.cuda_data_type(),
                shape.n as i32,
                ops.x.ptr as *const _,
                ops.x.dtype.cuda_data_type(),
                shape.k as i32,
                (&beta as *const f32).cast(),
                ops.c.ptr as *mut _,
                ops.c.dtype.cuda_data_type(),
                shape.n as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .expect("S3 native-half Fast GEMM");
    }

    fn modes(ctx: &GpuCtx) -> String {
        use cudarc::cublas::sys::*;
        let handle = *ctx.blas.handle();
        let success = cublasStatus_t::CUBLAS_STATUS_SUCCESS;
        let mut math = cublasMath_t::CUBLAS_DEFAULT_MATH;
        let mut pointer = cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST;
        let mut atomics = cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED;
        unsafe {
            assert_eq!(cublasSetMathMode(handle, math), success);
            assert_eq!(cublasSetPointerMode_v2(handle, pointer), success);
            assert_eq!(cublasSetAtomicsMode(handle, atomics), success);
            assert_eq!(cublasGetMathMode(handle, &mut math), success);
            assert_eq!(cublasGetPointerMode_v2(handle, &mut pointer), success);
            assert_eq!(cublasGetAtomicsMode(handle, &mut atomics), success);
        }
        assert_eq!(math, cublasMath_t::CUBLAS_DEFAULT_MATH);
        assert_eq!(pointer, cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        assert_eq!(atomics, cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED);
        format!(
            "\"compute\":\"CUBLAS_COMPUTE_32F\",\"algorithm\":\"CUBLAS_GEMM_DEFAULT_TENSOR_OP\",\"math\":\"{math:?}\",\"pointer_mode\":\"{pointer:?}\",\"atomics\":\"{atomics:?}\",\"bias_broadcast\":false"
        )
    }

    // Inspect every actual captured node. The existing one-node contract is
    // applied to each custom node, while the full graph count is checked here.
    fn inventory(
        graph: &CudaGraph,
        arm: usize,
        ops: InferenceFwdOperands,
        shape: InferenceShape,
        logical_ops: usize,
        vendor_nodes_per_op: usize,
    ) -> (String, usize) {
        use cudarc::driver::sys;
        let mut count = 0;
        assert_eq!(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) },
            sys::CUresult::CUDA_SUCCESS
        );
        assert!(count > 0);
        let mut nodes = vec![std::ptr::null_mut(); count];
        assert_eq!(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count) },
            sys::CUresult::CUDA_SUCCESS
        );
        if arm < 2 {
            assert_eq!(count, logical_ops);
        } else if vendor_nodes_per_op > 0 {
            assert_eq!(count, logical_ops * vendor_nodes_per_op);
        }
        let mut entries = Vec::new();
        for node in nodes {
            let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            assert_eq!(
                unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                kind,
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL,
                "no reset/copy/empty nodes in timed graph"
            );
            let mut p: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
            assert_eq!(
                unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut p) },
                sys::CUresult::CUDA_SUCCESS
            );
            let mut name = std::ptr::null();
            assert_eq!(
                unsafe { sys::cuFuncGetName(&mut name, p.func) },
                sys::CUresult::CUDA_SUCCESS
            );
            assert!(!name.is_null());
            let symbol = unsafe { CStr::from_ptr(name) }.to_str().unwrap();
            let mut decoded = String::new();
            if arm < 2 {
                let mut abi = Vec::new();
                for index in 0..5 {
                    let (mut offset, mut size) = (0, 0);
                    assert_eq!(
                        unsafe { sys::cuFuncGetParamInfo(p.func, index, &mut offset, &mut size) },
                        sys::CUresult::CUDA_SUCCESS
                    );
                    abi.push((offset, size));
                }
                let (mut offset, mut size) = (0, 0);
                let terminal =
                    unsafe { sys::cuFuncGetParamInfo(p.func, 5, &mut offset, &mut size) }
                        == sys::CUresult::CUDA_ERROR_INVALID_VALUE;
                assert!(!p.kernelParams.is_null());
                let mut pointers = [0u64; 4];
                for (index, value) in pointers.iter_mut().enumerate() {
                    let arg = unsafe { *p.kernelParams.add(index) };
                    assert!(!arg.is_null());
                    *value = unsafe { arg.cast::<u64>().read_unaligned() };
                }
                let bundle_ptr = unsafe { *p.kernelParams.add(4) };
                assert!(!bundle_ptr.is_null());
                let bundle = unsafe { bundle_ptr.cast::<[u32; 8]>().read_unaligned() };
                fixed_explicit_vendor_pipeline_graph_contract(
                    if arm == 0 {
                        InferenceTile::Tc128Sm89Swizzle
                    } else {
                        InferenceTile::Tc128Sm89S3
                    },
                    ops.x.dtype,
                    1,
                    ObservedGemmNode {
                        symbol,
                        grid: (p.gridDimX, p.gridDimY, p.gridDimZ),
                        block: (p.blockDimX, p.blockDimY, p.blockDimZ),
                        shared_bytes: p.sharedMemBytes,
                        driver_abi: abi,
                        terminal_sixth_rejected: terminal,
                        pointers,
                        bundle,
                    },
                    [ops.c.ptr, ops.x.ptr, ops.w.ptr, 0],
                    shape,
                )
                .unwrap();
                decoded = format!(
                    ",\"pointers\":{pointers:?},\"bundle\":{bundle:?},\"abi\":[[0,8],[8,8],[16,8],[24,8],[32,32]],\"sixth_rejected\":true"
                );
            }
            entries.push(format!("{{\"symbol\":\"{}\",\"grid\":[{},{},{}],\"block\":[{},{},{}],\"shared_bytes\":{}{} }}",
                fixed_sm120_tf32_bd_json_escape(symbol),p.gridDimX,p.gridDimY,p.gridDimZ,p.blockDimX,p.blockDimY,p.blockDimZ,p.sharedMemBytes,decoded));
        }
        (format!("[{}]", entries.join(",")), count)
    }

    pub(super) fn run(stage: Stage) {
        let controls = stage.controls();
        assert_eq!(
            std::env::var(controls[0]).as_deref(),
            Ok("1"),
            "explicit stage-specific S3 pair enable required"
        );
        if cfg!(debug_assertions) {
            panic!("S3 pairing requires release");
        }
        let environment: Vec<_> = std::env::vars_os()
            .map(|(key, _)| key.to_string_lossy().into_owned())
            .collect();
        validate_controls(stage, environment.iter().map(String::as_str)).unwrap();
        let dtype_text = std::env::var(controls[2]).expect("explicit dtypes");
        let (windows, dtypes) = config(
            stage,
            &std::env::var(controls[1]).expect("explicit windows"),
            &dtype_text,
        )
        .unwrap();
        fixed_sm120_tf32_bd_environment_preflight("Ada S3 mirrored paired").unwrap();
        let device = GpuDevice::new(0).expect("Ada device");
        assert_eq!(device.compute_capability, (8, 9));
        assert_eq!(device.multiprocessor_count(), 142);
        assert_eq!(TUNING_TABLE_REVISION, stage.revision());
        let ctx = GpuCtx::new(&device).expect("Ada context");
        let compiler = ctx.kernels.compiler_identity();
        assert!(compiler.nvrtc_library_known);
        let toolkit = std::env::var("S3_TOOLKIT").expect("toolkit binding");
        assert_eq!(
            toolkit,
            format!("{}.{}", compiler.nvrtc_version.0, compiler.nvrtc_version.1)
        );
        assert!(matches!(
            compiler.nvrtc_version,
            (12, 8) | (13, 0) | (13, 2)
        ));
        if stage == Stage::PostAuto43 {
            assert_eq!(
                compiler.nvrtc_version,
                (13, 2),
                "post-AUTO qualification is CUDA13.2-only"
            );
        }
        let artifact = ctx.kernels.artifact_set_identity().fixed;
        let source_hash = digest_hex(
            &Sha256::digest(std::fs::read("tests/gemm_bi_inference_performance.rs").unwrap())
                .into(),
        );
        let binary_hash = digest_hex(
            &Sha256::digest(std::fs::read(std::env::current_exe().unwrap()).unwrap()).into(),
        );
        assert_eq!(
            source_hash,
            std::env::var("S3_SOURCE_SHA").expect("source binding")
        );
        assert_eq!(
            binary_hash,
            std::env::var("S3_BINARY_SHA").expect("binary binding")
        );
        let mode = modes(&ctx);
        let arms = stage.arms();
        let directions = stage.ratios();
        let metadata = match stage {
            Stage::PrePromotion42 => format!(
                "\"schema\":\"MambaBiFixedAdaS3PairedV1\",\"toolkit\":\"{toolkit}\",\"shape\":[4621,768,2304],\"bias\":false,\"alpha\":1,\"beta\":0,\"revision\":42"
            ),
            Stage::PostAuto43 => format!(
                "\"schema\":\"MambaBiFixedAdaS3PostAutoPairedV1\",\"stage\":\"post_auto\",\"toolkit\":\"{toolkit}\",\"shape\":[4621,768,2304],\"bias\":false,\"alpha\":1,\"beta\":0,\"revision\":43"
            ),
        };
        println!(
            "{{{metadata},\"kind\":\"identity\",\"uuid\":\"GPU-d1edd7be-e88d-aed6-047d-622163306f0e\",\"cc\":\"8.9\",\"sm_count\":142,\"source_sha\":\"{source_hash}\",\"binary_sha\":\"{binary_hash}\",\"fixed_source_digest\":\"{}\",\"fixed_invocation_digest\":\"{}\",\"fixed_artifact_digest\":\"{}\",\"header_manifest_digest\":\"{}\",\"nvrtc_library_domain\":\"{}\",\"nvrtc_library_known\":true,\"dtypes\":\"{dtype_text}\",\"windows\":{windows},\"logical_ops\":20,\"warmup_eager\":128,\"percentile\":\"round((len-1)*fraction)\",{mode}}}",
            digest_hex(&compiler.source_digest),
            digest_hex(&compiler.invocation_digest),
            digest_hex(&artifact.artifact_digest),
            digest_hex(&compiler.header_manifest_digest),
            digest_hex(&compiler.nvrtc_library_domain)
        );
        let shape = InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        };
        let elements = shape.m * shape.n;
        let rows = fixed_explicit_vendor_row_specs();
        let mut configurations = 0;
        for index in dtypes {
            let row = rows[index];
            let dtype = row.input_dtype;
            let dtype_name = row.name;
            configure_fixed_auto_vendor_custom(&ctx, row.policy);
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).unwrap();
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).unwrap();
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x0ada_a001))
                .unwrap();
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x0ada_b001))
                .unwrap();
            let a_raw = fixed_explicit_vendor_raw_bytes(&ctx, &a);
            let b_raw = fixed_explicit_vendor_raw_bytes(&ctx, &b);
            let owners: Vec<_> = (0..3)
                .map(|_| {
                    let owner = DtypedBuf::zeros(&ctx.stream, elements + GUARD, dtype).unwrap();
                    upload(&ctx, owner.cached_ptr(), &vec![0x5a; owner.size_bytes()]);
                    owner
                })
                .collect();
            let operands: Vec<_> = owners
                .iter()
                .map(|owner| InferenceFwdOperands {
                    c: TypedPtr {
                        ptr: owner.cached_ptr() + GUARD as u64,
                        dtype,
                    },
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr: None,
                })
                .collect();
            assert!(operands.iter().all(|ops| ops.c.ptr % 256 == 0));
            let launch = |arm: usize| match arm {
                0 => {
                    if stage == Stage::PrePromotion42 {
                        let actual = launch_fixed_auto_vendor_custom(&ctx, operands[0], shape);
                        assert_eq!(
                            actual,
                            InferenceTile::Tc128Sm89Swizzle,
                            "historical actual AUTO must remain rev42 incumbent"
                        );
                    } else {
                        inference_forward_with_tile(
                            &ctx,
                            operands[0],
                            shape,
                            InferenceTile::Tc128Sm89Swizzle,
                        )
                        .expect("public forced Swizzle control");
                    }
                }
                1 => {
                    if stage == Stage::PrePromotion42 {
                        inference_forward_with_tile(
                            &ctx,
                            operands[1],
                            shape,
                            InferenceTile::Tc128Sm89S3,
                        )
                        .expect("public S3 force");
                    } else {
                        let actual = launch_fixed_auto_vendor_custom(&ctx, operands[1], shape);
                        assert_eq!(
                            actual,
                            InferenceTile::Tc128Sm89S3,
                            "post-AUTO must use actual public S3 selection"
                        );
                    }
                }
                2 => fast(&ctx, operands[2], shape),
                _ => unreachable!(),
            };
            for arm in 0..3 {
                launch(arm);
            }
            let expected: Vec<_> = owners.iter().map(|o| raw(&ctx, o)).collect();
            assert_eq!(
                expected[0], expected[1],
                "S3 and Swizzle exact homogeneous storage bits"
            );
            let reference = DtypedBuf::zeros(&ctx.stream, elements, WeightDtype::F32).unwrap();
            fixed_ada_vendor_launch(
                &ctx,
                InferenceFwdOperands {
                    c: typed(&reference, WeightDtype::F32),
                    ..operands[0]
                },
                shape,
                cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
            );
            let reference_bits = f32_bits(&ctx, &reference, elements);
            let errors: Vec<_> = expected
                .iter()
                .map(|bytes| {
                    fixed_ada_normalized_error(
                        &half_bits(bytes, dtype),
                        &reference_bits,
                        row.custom_tolerance,
                        "S3 paired numerical",
                    )
                })
                .collect();
            let mut one = Vec::new();
            let mut twenty = Vec::new();
            for arm in 0..3 {
                for _ in 0..2 {
                    assert!(
                        output_gate(&ctx, &owners[arm], &expected[arm], || launch(arm)),
                        "eager repeat storage gate"
                    );
                }
                let g = unsafe {
                    capture_into_graph(&ctx.stream, || {
                        launch(arm);
                        Ok(())
                    })
                }
                .unwrap();
                let (one_inventory, nodes) = inventory(&g, arm, operands[arm], shape, 1, 0);
                let g20 = unsafe {
                    capture_into_graph(&ctx.stream, || {
                        for _ in 0..20 {
                            launch(arm);
                        }
                        Ok(())
                    })
                }
                .unwrap();
                let (twenty_inventory, _) = inventory(&g20, arm, operands[arm], shape, 20, nodes);
                for graph in [&g, &g20] {
                    for _ in 0..2 {
                        assert!(
                            output_gate(&ctx, &owners[arm], &expected[arm], || graph
                                .launch()
                                .unwrap()),
                            "independent graph overwrite/bits gate"
                        );
                    }
                }
                println!(
                    "{{{metadata},\"kind\":\"physical\",\"dtype\":\"{dtype_name}\",\"arm\":\"{}\",\"pointers\":[{},{},{},0],\"allocation\":[{},{}],\"guard_bytes\":256,\"one\":{one_inventory},\"twenty\":{twenty_inventory},\"numerical_error\":{},\"tolerance\":{},\"reference\":\"PEDANTIC_F32\",\"eager_repeats\":2,\"graph_repeats\":2,\"poison_upload_verified\":true,\"repeat_bits\":true,\"guards\":true}}",
                    arms[arm],
                    operands[arm].c.ptr,
                    operands[arm].x.ptr,
                    operands[arm].w.ptr,
                    owners[arm].cached_ptr(),
                    owners[arm].size_bytes(),
                    errors[arm],
                    row.custom_tolerance
                );
                one.push(g);
                twenty.push(g20);
            }
            let noop = unsafe { capture_into_graph(&ctx.stream, || Ok(())) }
                .expect("empty negative graph capture");
            assert!(
                !output_gate(&ctx, &owners[1], &expected[1], || noop.launch().unwrap()),
                "no-op candidate must fail overwrite/bits gate"
            );
            for path in ["eager", "graph"] {
                for start in 0..2 {
                    for arm in 0..3 {
                        assert!(output_gate(&ctx, &owners[arm], &expected[arm], || launch(
                            arm
                        )));
                        for graph in [&one[arm], &twenty[arm]] {
                            assert!(output_gate(&ctx, &owners[arm], &expected[arm], || graph
                                .launch()
                                .unwrap()));
                        }
                    }
                    unchanged_inputs(
                        &a_raw,
                        &b_raw,
                        &fixed_explicit_vendor_raw_bytes(&ctx, &a),
                        &fixed_explicit_vendor_raw_bytes(&ctx, &b),
                    )
                    .expect("pre-timing immutable input gate");
                    // The required input readback gate is the final action before
                    // timing preparation. After capacity is reserved, nothing prints,
                    // downloads, poisons, compiles or allocates device memory until
                    // all windows of this configuration have completed.
                    let schedule: Vec<_> = (0..windows)
                        .flat_map(|w| schedule(w, start).into_iter().map(move |s| (w, s)))
                        .collect();
                    let mut samples = Vec::with_capacity(12 * windows);
                    for _ in 0..128 {
                        for arm in 0..3 {
                            launch(arm);
                        }
                    }
                    for arm in 0..3 {
                        one[arm].launch().unwrap();
                        twenty[arm].launch().unwrap();
                    }
                    ctx.stream.synchronize().unwrap();
                    for &(window, (traversal, comparison, position, arm)) in &schedule {
                        let us = if path == "eager" {
                            fixed_ada_event_window_us(&ctx, 20, || launch(arm))
                        } else {
                            fixed_ada_event_window_us(&ctx, 1, || twenty[arm].launch().unwrap())
                                / 20.0
                        };
                        samples.push((window, traversal, comparison, position, arm, us));
                    }
                    for arm in 0..3 {
                        assert_eq!(
                            raw(&ctx, &owners[arm]),
                            expected[arm],
                            "post timing exact bits"
                        );
                        assert!(output_gate(&ctx, &owners[arm], &expected[arm], || launch(
                            arm
                        )));
                        for graph in [&one[arm], &twenty[arm]] {
                            assert!(output_gate(&ctx, &owners[arm], &expected[arm], || graph
                                .launch()
                                .unwrap()));
                        }
                    }
                    assert_eq!(
                        fixed_explicit_vendor_raw_bytes(&ctx, &a),
                        a_raw,
                        "immutable A"
                    );
                    assert_eq!(
                        fixed_explicit_vendor_raw_bytes(&ctx, &b),
                        b_raw,
                        "immutable B"
                    );
                    let key = format!(
                        "{metadata},\"dtype\":\"{dtype_name}\",\"path\":\"{path}\",\"start_parity\":{start}"
                    );
                    let mut ratios: [Vec<f64>; 3] =
                        std::array::from_fn(|_| Vec::with_capacity(windows));
                    for (chronology, &(w, traversal, comparison, position, arm, us)) in
                        samples.iter().enumerate()
                    {
                        let order = if (w + start) % 2 == 0 { "ABBA" } else { "BAAB" };
                        println!(
                            "{{{key},\"kind\":\"sample\",\"chronology\":{chronology},\"window\":{w},\"comparison\":{comparison},\"traversal\":{traversal},\"order\":\"{order}\",\"position\":{position},\"arm\":\"{}\",\"logical_ops\":20,\"us\":{us}}}",
                            arms[arm]
                        );
                    }
                    for (bracket, observations) in samples.as_chunks::<4>().0.iter().enumerate() {
                        let (w, traversal, comparison, _, _, _) = observations[0];
                        let pair: Vec<_> = observations.iter().map(|o| (o.4, o.5)).collect();
                        let ratio = pair_ratio(&pair, comparison).unwrap();
                        ratios[comparison].push(ratio);
                        println!(
                            "{{{key},\"kind\":\"pair\",\"window\":{w},\"comparison\":{comparison},\"traversal\":{traversal},\"observations\":[{},{},{},{}],\"ratio\":{ratio}}}",
                            bracket * 4,
                            bracket * 4 + 1,
                            bracket * 4 + 2,
                            bracket * 4 + 3
                        );
                    }
                    for comparison in 0..3 {
                        ratios[comparison].sort_by(f64::total_cmp);
                        let p50 = percentile(&ratios[comparison], 0.5);
                        let p95 = percentile(&ratios[comparison], 0.95);
                        println!(
                            "{{{key},\"kind\":\"summary\",\"comparison\":{comparison},\"direction\":\"{}\",\"windows\":{windows},\"p50\":{p50},\"p95\":{p95}}}",
                            directions[comparison]
                        );
                    }
                    println!(
                        "{{{key},\"kind\":\"configuration_complete\",\"samples\":{},\"pairs\":{},\"summaries\":3,\"pre_post_bits\":true,\"pre_post_graphs\":true,\"guards\":true,\"immutable_inputs\":true,\"noop_rejected\":true}}",
                        12 * windows,
                        3 * windows
                    );
                    configurations += 1;
                }
            }
        }
        println!(
            "{{{metadata},\"kind\":\"complete\",\"configurations\":{configurations},\"samples\":{},\"pairs\":{},\"summaries\":{},\"rejected\":0,\"passed\":true}}",
            configurations * 12 * windows,
            configurations * 3 * windows,
            configurations * 3
        );
    }

    #[test]
    fn mirrored_brackets_reverse_both_arm_order_and_comparison_traversal() {
        assert_eq!(
            schedule(0, 0),
            vec![
                (0, 0, 0, 0),
                (0, 0, 1, 1),
                (0, 0, 2, 1),
                (0, 0, 3, 0),
                (1, 1, 0, 2),
                (1, 1, 1, 0),
                (1, 1, 2, 0),
                (1, 1, 3, 2),
                (2, 2, 0, 2),
                (2, 2, 1, 1),
                (2, 2, 2, 1),
                (2, 2, 3, 2)
            ]
        );
        assert_eq!(
            schedule(0, 1),
            vec![
                (0, 2, 0, 1),
                (0, 2, 1, 2),
                (0, 2, 2, 2),
                (0, 2, 3, 1),
                (1, 1, 0, 0),
                (1, 1, 1, 2),
                (1, 1, 2, 2),
                (1, 1, 3, 0),
                (2, 0, 0, 1),
                (2, 0, 1, 0),
                (2, 0, 2, 0),
                (2, 0, 3, 1)
            ]
        );
        assert_eq!(schedule(1, 0), schedule(0, 1));
        assert_eq!(schedule(1, 1), schedule(0, 0));
    }
}

// Separate protocol: do not reinterpret the historical triple comparator below.
#[test]
#[ignore = "requires explicit MAMBA_FIXED_ADA_S3_PAIR=1 and exclusive pinned Ada; production AUTO/S3/Fast mirrored brackets"]
fn fixed_ada_half_s3_auto_fast_paired() {
    ada_s3_pair::run(ada_s3_pair::Stage::PrePromotion42);
}

#[test]
#[ignore = "requires explicit MAMBA_FIXED_ADA_S3_POST_PAIR=1 and exclusive pinned Ada CUDA13.2; forced Swizzle/actual AUTO43/Fast mirrored brackets"]
fn fixed_ada_half_s3_post_auto_fast_paired() {
    ada_s3_pair::run(ada_s3_pair::Stage::PostAuto43);
}

#[path = "support/fixed_sm89_exact_n64_admission.rs"]
mod exact_n64_admission;

#[path = "support/fixed_sm89_toolkit_admission.rs"]
mod toolkit_admission;

#[test]
#[ignore = "requires exclusive pinned Ada, actual NVRTC candidate admission and paired evidence output"]
fn fixed_sm89_exact_n64_paired_admission() {
    exact_n64_admission::run();
}

fn typed(buffer: &DtypedBuf, dtype: WeightDtype) -> TypedPtr {
    TypedPtr {
        ptr: buffer.cached_ptr(),
        dtype,
    }
}

fn average_us(ctx: &GpuCtx, mut run: impl FnMut()) -> f64 {
    for _ in 0..WARMUPS {
        run();
    }
    ctx.stream.synchronize().expect("warmup sync");
    let started = Instant::now();
    for _ in 0..ITERS {
        run();
    }
    ctx.stream.synchronize().expect("timing sync");
    started.elapsed().as_secs_f64() * 1e6 / ITERS as f64
}

fn fixed_tile_window_us(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    tile: InferenceTile,
    iterations: usize,
) -> f64 {
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record Fixed window start");
    for _ in 0..iterations {
        inference_forward_with_tile(ctx, operands, shape, tile)
            .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record Fixed window end");
    f64::from(start.elapsed_ms(&end).expect("measure Fixed window")) * 1000.0 / iterations as f64
}

fn fixed_tile_window_iterations(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    tile: InferenceTile,
) -> usize {
    let pilot_iterations = 16;
    let pilot_us = fixed_tile_window_us(ctx, operands, shape, tile, pilot_iterations);
    (5000.0 / pilot_us).ceil().clamp(1.0, 4096.0) as usize
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

fn synth(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state & 0xffff) as f32 / 32768.0) - 1.0
        })
        .collect()
}

fn f32_bits(ctx: &GpuCtx, buffer: &DtypedBuf, len: usize) -> Vec<u32> {
    let mut host = vec![0.0f32; len];
    buffer
        .download_f32(&ctx.stream, &mut host)
        .expect("f32 result download");
    host.into_iter().map(f32::to_bits).collect()
}

fn single_graph_kernel_name(graph: &CudaGraph, label: &str) -> String {
    let mut node_count = 0;
    assert_eq!(
        unsafe {
            cudarc::driver::sys::cuGraphGetNodes(
                graph.cu_graph(),
                std::ptr::null_mut(),
                &mut node_count,
            )
        },
        cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "{label} graph node count"
    );
    assert_eq!(node_count, 1, "{label} physical graph node inventory");
    let mut nodes = vec![std::ptr::null_mut(); node_count];
    assert_eq!(
        unsafe {
            cudarc::driver::sys::cuGraphGetNodes(
                graph.cu_graph(),
                nodes.as_mut_ptr(),
                &mut node_count,
            )
        },
        cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "{label} graph nodes"
    );
    let mut params = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { cudarc::driver::sys::cuGraphKernelNodeGetParams_v2(nodes[0], &mut params) },
        cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "{label} graph kernel params"
    );
    let mut function_name = std::ptr::null();
    assert_eq!(
        unsafe { cudarc::driver::sys::cuFuncGetName(&mut function_name, params.func) },
        cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "{label} graph kernel name"
    );
    unsafe { CStr::from_ptr(function_name) }
        .to_str()
        .expect("UTF-8 CUDA function name")
        .to_owned()
}

#[test]
fn fixed_sm120_copyplan_t256_spike_mapping_and_graph_geometry() {
    let control = fixed_sm120_fma_spike_tile(Some("copyplan")).unwrap();
    assert_eq!(control, InferenceTile::F32Sm120N64CopyPlan);
    assert_eq!(sm120_exact_tma_graph_geometry(control), (64, 64, 128, 0));
    let wide = fixed_sm120_fma_spike_tile(Some("copyplan_m128n64_t256")).unwrap();
    assert_eq!(wide, InferenceTile::F32Sm120M128N64CopyPlanT256);
    assert_eq!(sm120_exact_tma_graph_geometry(wide), (128, 64, 256, 0));
    let tile = fixed_sm120_fma_spike_tile(Some("copyplan_t256")).unwrap();
    assert_eq!(tile, InferenceTile::F32Sm120N64CopyPlanT256);
    assert_eq!(sm120_exact_tma_graph_geometry(tile), (64, 64, 256, 0));
}

#[test]
fn fixed_sm120_nobias_t256_spike_mapping_and_graph_geometry() {
    let tile = fixed_sm120_fma_spike_tile(Some("nobias_m128n64_t256")).unwrap();
    assert_eq!(tile, InferenceTile::F32Sm120TmaFmaFixedNoBiasM128N64T256);
    assert_eq!(sm120_exact_tma_graph_geometry(tile), (128, 64, 256, 24_592));
    assert_eq!(
        fixed_sm120_fma_spike_tile(Some("postbias_m128n64_t256")).unwrap(),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
    );
    assert!(fixed_sm120_fma_spike_tile(Some("nobias_m128n64")).is_err());
}

fn sm120_exact_tma_graph_geometry(tile: InferenceTile) -> (usize, usize, u32, u32) {
    match tile {
        InferenceTile::F32Sm120N64CopyPlan => (64, 64, 128, 0),
        InferenceTile::F32Sm120N64CopyPlanT256 => (64, 64, 256, 0),
        InferenceTile::F32Sm120M128N64CopyPlanT256 => (128, 64, 256, 0),
        InferenceTile::F32Sm120TmaFmaFixedNoBiasM128N64T256
        | InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256 => (128, 64, 256, 24_592),
        InferenceTile::F32Sm120TmaFmaM128N64
        | InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64
        | InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64K4 => (128, 64, 128, 24_592),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96 => (128, 96, 256, 28_688),
        InferenceTile::F32Sm120TmaFmaM64N128
        | InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128 => (64, 128, 128, 24_592),
        _ => panic!("non-TMA tile in exact-TMA graph contract: {tile:?}"),
    }
}

fn assert_sm120_exact_tma_graph_contract(
    graph: &CudaGraph,
    output: u64,
    shape: InferenceShape,
    bias: Option<u64>,
    tile: InferenceTile,
) {
    use cudarc::driver::sys;

    let mut node_count = 0usize;
    assert_eq!(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut node_count) },
        sys::CUresult::CUDA_SUCCESS
    );
    assert_eq!(node_count, 1);
    let mut nodes = vec![std::ptr::null_mut(); node_count];
    assert_eq!(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut node_count) },
        sys::CUresult::CUDA_SUCCESS
    );
    let mut params: sys::CUDA_KERNEL_NODE_PARAMS_v2 = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { sys::cuGraphKernelNodeGetParams_v2(nodes[0], &mut params) },
        sys::CUresult::CUDA_SUCCESS
    );
    let (tile_m, tile_n, threads, dynamic_shared) = sm120_exact_tma_graph_geometry(tile);
    assert_eq!(
        (params.gridDimX, params.gridDimY, params.gridDimZ),
        (
            (shape.m.div_ceil(tile_m) * shape.n.div_ceil(tile_n)) as u32,
            1,
            1
        )
    );
    assert_eq!(
        (params.blockDimX, params.blockDimY, params.blockDimZ),
        (threads, 1, 1)
    );
    assert_eq!(params.sharedMemBytes, dynamic_shared);
    if matches!(
        tile,
        InferenceTile::F32Sm120N64CopyPlan
            | InferenceTile::F32Sm120N64CopyPlanT256
            | InferenceTile::F32Sm120M128N64CopyPlanT256
    ) {
        let mut layout = Vec::new();
        for index in 0..5 {
            let mut offset = 0;
            let mut size = 0;
            assert_eq!(
                unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) },
                sys::CUresult::CUDA_SUCCESS
            );
            layout.push((offset, size));
        }
        assert_eq!(layout, vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]);
        let mut offset = 0;
        let mut size = 0;
        assert_eq!(
            unsafe { sys::cuFuncGetParamInfo(params.func, 5, &mut offset, &mut size) },
            sys::CUresult::CUDA_ERROR_INVALID_VALUE
        );
        let captured = |index: usize| unsafe { *params.kernelParams.add(index) };
        let pointer = |index| unsafe { *(captured(index) as *const u64) };
        assert_eq!(pointer(0), output);
        assert_ne!(pointer(1), 0);
        assert_ne!(pointer(2), 0);
        assert_eq!(pointer(3), bias.unwrap_or(0));
        assert_eq!(
            unsafe { *(captured(4) as *const [u32; 8]) },
            [
                1.0f32.to_bits(),
                0,
                shape.m as u32,
                shape.n as u32,
                shape.k as u32,
                shape.k as u32,
                shape.n as u32,
                shape.n as u32
            ]
        );
        return;
    }
    let mut layout = Vec::new();
    for index in 0..7 {
        let mut offset = 0usize;
        let mut size = 0usize;
        assert_eq!(
            unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) },
            sys::CUresult::CUDA_SUCCESS
        );
        layout.push((offset, size));
    }
    assert_eq!(
        layout,
        vec![
            (0, 8),
            (8, 8),
            (16, 8),
            (128, 128),
            (256, 128),
            (384, 8),
            (392, 32),
        ]
    );
    let captured = |index: usize| unsafe { *params.kernelParams.add(index) };
    let pointer = |index: usize| unsafe { *(captured(index) as *const u64) };
    assert_eq!(pointer(0), output);
    assert_eq!(
        pointer(1),
        0,
        "single-split exact TMA must not bind scratch"
    );
    assert_eq!(pointer(2), 0, "single-split exact TMA must not bind flags");
    assert!(
        unsafe { *(captured(3) as *const [u8; 128]) }
            .iter()
            .any(|byte| *byte != 0)
    );
    assert!(
        unsafe { *(captured(4) as *const [u8; 128]) }
            .iter()
            .any(|byte| *byte != 0)
    );
    assert_eq!(pointer(5), bias.unwrap_or(0));
    assert_eq!(
        unsafe { *(captured(6) as *const [u32; 8]) },
        [
            1.0f32.to_bits(),
            0,
            shape.m as u32,
            shape.n as u32,
            shape.k as u32,
            shape.n as u32,
            1,
            shape.k.div_ceil(16) as u32,
        ]
    );
}

#[test]
fn forced_fixed_launch_uses_structured_arguments() {
    let _: fn(&GpuCtx, InferenceFwdOperands, InferenceShape, InferenceTile) -> Result<(), String> =
        inference_forward_with_tile;
}

#[test]
#[ignore = "requires a CC12.0 170-SM CUDA device"]
fn fixed_sm120_tf32_production_routes_match_forced_bits_and_graphs() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let m64n128 = &ctx
        .kernels
        .gemm_bi_nn_tf32_sm120
        .as_ref()
        .expect("SM120 TF32 kernels")
        .m64n128_s2;
    assert_eq!(
        m64n128
            .local_size_bytes()
            .expect("M64N128 local-memory footprint"),
        0,
        "SM120 TF32 M64N128 spills to local memory"
    );
    assert_eq!(
        m64n128
            .occupancy_max_active_blocks_per_multiprocessor(128, 49_280, None)
            .expect("M64N128 occupancy"),
        2,
        "SM120 TF32 M64N128 must retain two resident CTAs"
    );
    let m128n64 = &ctx
        .kernels
        .gemm_bi_nn_tf32_sm120
        .as_ref()
        .expect("SM120 TF32 kernels")
        .m128n64_s2;
    assert_eq!(
        m128n64
            .local_size_bytes()
            .expect("M128N64 local-memory footprint"),
        0,
        "SM120 TF32 M128N64 spills to local memory"
    );
    assert_eq!(
        m128n64
            .occupancy_max_active_blocks_per_multiprocessor(128, 49_280, None)
            .expect("M128N64 occupancy"),
        2,
        "SM120 TF32 M128N64 must retain two resident CTAs"
    );
    let (nvrtc_major, nvrtc_minor) = ctx.kernels.compiler_identity().nvrtc_version;
    configure_fixed_auto_vendor_custom(&ctx, F32TriadPolicy::AllowDeterministicTf32);
    for (cell, shape, output_offset, has_bias) in [
        (
            "A",
            InferenceShape {
                m: 4621,
                k: 384,
                n: 1928,
            },
            0,
            false,
        ),
        (
            "B",
            InferenceShape {
                m: 4621,
                k: 768,
                n: 2304,
            },
            0,
            false,
        ),
        (
            "C",
            InferenceShape {
                m: 4621,
                k: 1928,
                n: 384,
            },
            0,
            false,
        ),
        (
            "D",
            InferenceShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
            0,
            false,
        ),
        (
            "E",
            InferenceShape {
                m: 2048,
                k: 2304,
                n: 768,
            },
            0,
            false,
        ),
        (
            "A_bias",
            InferenceShape {
                m: 4621,
                k: 384,
                n: 1928,
            },
            0,
            true,
        ),
        (
            "B_bias",
            InferenceShape {
                m: 4621,
                k: 768,
                n: 2304,
            },
            0,
            true,
        ),
        (
            "D_bias",
            InferenceShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
            0,
            true,
        ),
        (
            "A_misaligned",
            InferenceShape {
                m: 4621,
                k: 384,
                n: 1928,
            },
            1,
            false,
        ),
        (
            "D_misaligned",
            InferenceShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
            1,
            false,
        ),
    ] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("production A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("production B allocation");
        let bias = DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32)
            .expect("production bias allocation");
        let output_len = output_offset + shape.m * shape.n;
        let forced = DtypedBuf::zeros(&ctx.stream, output_len, WeightDtype::F32)
            .expect("forced output allocation");
        let production = DtypedBuf::zeros(&ctx.stream, output_len, WeightDtype::F32)
            .expect("production output allocation");
        a.upload_f32(
            &ctx.stream,
            &synth(shape.m * shape.k, 0xa170_5052_4f44_0001),
        )
        .expect("production A upload");
        b.upload_f32(
            &ctx.stream,
            &synth(shape.k * shape.n, 0xb170_5052_4f44_0002),
        )
        .expect("production B upload");
        bias.upload_f32(&ctx.stream, &synth(shape.n, 0xb1a5_5052_4f44_0003))
            .expect("production bias upload");
        let bias_ptr = has_bias.then(|| bias.cached_ptr());
        let forced_operands = InferenceFwdOperands {
            c: TypedPtr {
                ptr: forced.cached_ptr() + (output_offset * std::mem::size_of::<f32>()) as u64,
                dtype: WeightDtype::F32,
            },
            x: typed(&a, WeightDtype::F32),
            w: typed(&b, WeightDtype::F32),
            bias_ptr,
        };
        let production_operands = InferenceFwdOperands {
            c: TypedPtr {
                ptr: production.cached_ptr() + (output_offset * std::mem::size_of::<f32>()) as u64,
                dtype: WeightDtype::F32,
            },
            ..forced_operands
        };
        inference_forward_with_tile(&ctx, forced_operands, shape, InferenceTile::Tf32Sm120M64S2)
            .expect("forced incumbent launch");
        let selected = inference_forward(
            &ctx,
            production_operands.c,
            production_operands.x,
            production_operands.w,
            bias_ptr,
            (shape.m, shape.k, shape.n),
        )
        .expect("production pair-store launch");
        let dims = (shape.m, shape.k, shape.n);
        let uses_wide_tile = (nvrtc_major, nvrtc_minor) == (13, 2)
            && (dims == (4621, 768, 2304)
                || (dims == (4621, 384, 1928)
                    && ctx.kernels.compiler_identity().nvrtc_library_known
                    && production_operands.c.ptr.is_multiple_of(8)));
        let uses_d_schedule = (nvrtc_major, nvrtc_minor) == (13, 2) && dims == (2048, 768, 2304);
        let uses_d_pair_store = uses_d_schedule && production_operands.c.ptr.is_multiple_of(8);
        let uses_d_producer_warp = uses_d_schedule && !uses_d_pair_store;
        let expected_tile = if uses_wide_tile {
            InferenceTile::Tf32Sm120M128S2
        } else if uses_d_pair_store {
            InferenceTile::Tf32Sm120M64S2PairStore
        } else if uses_d_producer_warp {
            InferenceTile::Tf32Sm120M64S2ProducerWarp
        } else {
            InferenceTile::Tf32Sm120M64S2
        };
        assert_eq!(selected, expected_tile, "{cell} route identity");
        ctx.stream.synchronize().expect("production eager sync");
        let reference = f32_bits(&ctx, &forced, output_len);
        assert_eq!(
            f32_bits(&ctx, &production, output_len),
            reference,
            "{cell} production bits"
        );
        let graph = unsafe {
            capture_into_graph(&ctx.stream, || {
                inference_forward(
                    &ctx,
                    production_operands.c,
                    production_operands.x,
                    production_operands.w,
                    bias_ptr,
                    (shape.m, shape.k, shape.n),
                )
                .map(|_| ())
            })
        }
        .expect("capture production pair-store graph");
        let mut node_count = 0;
        assert_eq!(
            unsafe {
                cudarc::driver::sys::cuGraphGetNodes(
                    graph.cu_graph(),
                    std::ptr::null_mut(),
                    &mut node_count,
                )
            },
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "{cell} graph node count"
        );
        assert_eq!(node_count, 1, "{cell} physical graph node inventory");
        let mut nodes = vec![std::ptr::null_mut(); node_count];
        assert_eq!(
            unsafe {
                cudarc::driver::sys::cuGraphGetNodes(
                    graph.cu_graph(),
                    nodes.as_mut_ptr(),
                    &mut node_count,
                )
            },
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "{cell} graph nodes"
        );
        let mut params = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { cudarc::driver::sys::cuGraphKernelNodeGetParams_v2(nodes[0], &mut params) },
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "{cell} graph kernel params"
        );
        let mut function_name = std::ptr::null();
        assert_eq!(
            unsafe { cudarc::driver::sys::cuFuncGetName(&mut function_name, params.func) },
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "{cell} graph kernel name"
        );
        let function_name = unsafe { CStr::from_ptr(function_name) }
            .to_str()
            .expect("UTF-8 CUDA function name");
        let uses_pair_store = output_offset == 0
            && ((nvrtc_major, nvrtc_minor) == (13, 0)
                || ((nvrtc_major, nvrtc_minor) == (12, 8)
                    && matches!(dims, (2048, 768, 2304) | (2048, 2304, 768))));
        let expected_function = if uses_wide_tile {
            "nn_sm120_tma_tf32_m128n64_bk32_s2"
        } else if uses_d_pair_store {
            "nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store"
        } else if uses_d_producer_warp {
            "nn_sm120_tma_tf32_m64n64_bk32_s2_producer_warp"
        } else if uses_pair_store {
            "nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store"
        } else {
            "nn_sm120_tma_tf32_m64n64_bk32_s2"
        };
        assert_eq!(function_name, expected_function, "{cell} physical route");
        if uses_wide_tile {
            assert_eq!(
                (params.gridDimX, params.gridDimY, params.gridDimZ),
                (
                    if dims == (4621, 384, 1928) {
                        1147
                    } else {
                        1332
                    },
                    1,
                    1
                )
            );
            assert_eq!(
                (params.blockDimX, params.blockDimY, params.blockDimZ),
                (128, 1, 1)
            );
            assert_eq!(params.sharedMemBytes, 49_280);
        }
        if uses_d_schedule {
            assert_eq!(
                (params.gridDimX, params.gridDimY, params.gridDimZ),
                (1152, 1, 1)
            );
            assert_eq!(
                (params.blockDimX, params.blockDimY, params.blockDimZ),
                (if uses_d_pair_store { 128 } else { 160 }, 1, 1)
            );
            assert_eq!(params.sharedMemBytes, 32_896);
        }
        if cell == "D" {
            let forced_graph = unsafe {
                capture_into_graph(&ctx.stream, || {
                    inference_forward_with_tile(
                        &ctx,
                        forced_operands,
                        shape,
                        InferenceTile::Tf32Sm120M64S2,
                    )
                })
            }
            .expect("capture forced incumbent graph");
            let mut forced_node_count = 0;
            assert_eq!(
                unsafe {
                    cudarc::driver::sys::cuGraphGetNodes(
                        forced_graph.cu_graph(),
                        std::ptr::null_mut(),
                        &mut forced_node_count,
                    )
                },
                cudarc::driver::sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(forced_node_count, 1, "forced physical graph node inventory");
            let mut forced_nodes = vec![std::ptr::null_mut(); forced_node_count];
            assert_eq!(
                unsafe {
                    cudarc::driver::sys::cuGraphGetNodes(
                        forced_graph.cu_graph(),
                        forced_nodes.as_mut_ptr(),
                        &mut forced_node_count,
                    )
                },
                cudarc::driver::sys::CUresult::CUDA_SUCCESS
            );
            let mut forced_params = unsafe { std::mem::zeroed() };
            assert_eq!(
                unsafe {
                    cudarc::driver::sys::cuGraphKernelNodeGetParams_v2(
                        forced_nodes[0],
                        &mut forced_params,
                    )
                },
                cudarc::driver::sys::CUresult::CUDA_SUCCESS
            );
            let mut forced_function_name = std::ptr::null();
            assert_eq!(
                unsafe {
                    cudarc::driver::sys::cuFuncGetName(
                        &mut forced_function_name,
                        forced_params.func,
                    )
                },
                cudarc::driver::sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                unsafe { CStr::from_ptr(forced_function_name) }
                    .to_str()
                    .expect("forced UTF-8 CUDA function name"),
                "nn_sm120_tma_tf32_m64n64_bk32_s2"
            );
        }
        for replay in 0..10 {
            graph.launch().expect("production graph launch");
            ctx.stream.synchronize().expect("production graph sync");
            assert_eq!(
                f32_bits(&ctx, &production, output_len),
                reference,
                "{cell} production graph replay {replay}"
            );
        }
    }
}

#[test]
fn f32_legacy_baseline_launch_uses_structured_arguments() {
    let _: fn(&GpuCtx, InferenceFwdOperands, InferenceShape) -> Result<(), String> =
        inference_forward_f32_legacy_baseline;
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn f32_s2_production_matches_legacy_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for shape in [
        InferenceShape { m: 1, k: 0, n: 1 },
        InferenceShape {
            m: 63,
            k: 31,
            n: 63,
        },
        InferenceShape {
            m: 64,
            k: 32,
            n: 64,
        },
        InferenceShape {
            m: 65,
            k: 33,
            n: 65,
        },
        InferenceShape {
            m: 65,
            k: 36,
            n: 68,
        },
        InferenceShape {
            m: 129,
            k: 97,
            n: 193,
        },
        InferenceShape {
            m: 4621,
            k: 384,
            n: 384,
        },
    ] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("B allocation");
        let baseline = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("baseline allocation");
        let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("candidate allocation");
        let bias =
            DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa11ce))
            .expect("A upload");
        b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb0b))
            .expect("B upload");
        bias.upload_f32(&ctx.stream, &synth(shape.n, 0xb1a5))
            .expect("bias upload");
        for bias_ptr in [None, Some(bias.cached_ptr())] {
            let baseline_operands = InferenceFwdOperands {
                c: typed(&baseline, WeightDtype::F32),
                x: typed(&a, WeightDtype::F32),
                w: typed(&b, WeightDtype::F32),
                bias_ptr,
            };
            let candidate_operands = InferenceFwdOperands {
                c: typed(&candidate, WeightDtype::F32),
                ..baseline_operands
            };
            inference_forward_f32_legacy_baseline(&ctx, baseline_operands, shape)
                .expect("legacy Fixed launch");
            let tile = inference_forward(
                &ctx,
                candidate_operands.c,
                candidate_operands.x,
                candidate_operands.w,
                bias_ptr,
                (shape.m, shape.k, shape.n),
            )
            .expect("production S2 launch");
            assert_eq!(tile, InferenceTile::Legacy);
            ctx.stream.synchronize().expect("S2 sync");
            assert_eq!(
                f32_bits(&ctx, &candidate, shape.m * shape.n),
                f32_bits(&ctx, &baseline, shape.m * shape.n),
                "F32 production S2 changed legacy bits for M{} K{} N{} bias={}",
                shape.m,
                shape.k,
                shape.n,
                bias_ptr.is_some(),
            );
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn fixed_tf32_forward_is_repeatable_and_tile_invariant() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32);
    let tiles = [
        InferenceTile::Tf32M128S2,
        InferenceTile::Tf32M128S3,
        InferenceTile::Tf32M64S2,
        InferenceTile::Tf32M64S3,
        InferenceTile::Tf32M16S4,
    ];
    for shape in [
        InferenceShape { m: 1, k: 0, n: 1 },
        InferenceShape {
            m: 15,
            k: 31,
            n: 31,
        },
        InferenceShape {
            m: 65,
            k: 36,
            n: 68,
        },
        InferenceShape {
            m: 129,
            k: 97,
            n: 193,
        },
    ] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("B allocation");
        let bias =
            DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x7f32a))
            .expect("A upload");
        b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x7f32b))
            .expect("B upload");
        bias.upload_f32(&ctx.stream, &synth(shape.n, 0x7f32c))
            .expect("bias upload");
        for bias_ptr in [None, Some(bias.cached_ptr())] {
            let mut reference = None;
            for tile in tiles {
                let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                    .expect("output allocation");
                let operands = InferenceFwdOperands {
                    c: typed(&output, WeightDtype::F32),
                    x: typed(&a, WeightDtype::F32),
                    w: typed(&b, WeightDtype::F32),
                    bias_ptr,
                };
                inference_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                ctx.stream.synchronize().expect("first TF32 sync");
                let first = f32_bits(&ctx, &output, shape.m * shape.n);
                inference_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("repeated {tile:?}: {error}"));
                ctx.stream.synchronize().expect("second TF32 sync");
                let second = f32_bits(&ctx, &output, shape.m * shape.n);
                assert_eq!(first, second, "TF32 repeat drift for {shape:?} {tile:?}");
                if let Some(expected) = &reference {
                    assert_eq!(
                        &first,
                        expected,
                        "TF32 tile drift for {shape:?} {tile:?} bias={}",
                        bias_ptr.is_some()
                    );
                } else {
                    reference = Some(first);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_tf32_is_portable_bit_exact_and_selected() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    if !matches!(
        ctx.stream
            .context()
            .compute_capability()
            .expect("CUDA compute capability"),
        (12, 0) | (12, 1)
    ) {
        return;
    }
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32);
    let specialized = [
        InferenceTile::Tf32Sm120M128S2,
        InferenceTile::Tf32Sm120M128S3,
        InferenceTile::Tf32Sm120M64N128S2,
        InferenceTile::Tf32Sm120M64N128S3,
        InferenceTile::Tf32Sm120M64S2ProducerWarp,
        InferenceTile::Tf32Sm120M64S2,
    ];
    for shape in [
        InferenceShape {
            m: 65,
            k: 36,
            n: 68,
        },
        InferenceShape {
            m: 129,
            k: 96,
            n: 196,
        },
    ] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("B allocation");
        let bias =
            DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x120a))
            .expect("A upload");
        b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x120b))
            .expect("B upload");
        bias.upload_f32(&ctx.stream, &synth(shape.n, 0x120c))
            .expect("bias upload");
        for bias_ptr in [None, Some(bias.cached_ptr())] {
            let portable = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("portable output");
            let portable_operands = InferenceFwdOperands {
                c: typed(&portable, WeightDtype::F32),
                x: typed(&a, WeightDtype::F32),
                w: typed(&b, WeightDtype::F32),
                bias_ptr,
            };
            inference_forward_with_tile(&ctx, portable_operands, shape, InferenceTile::Tf32M64S2)
                .expect("portable TF32 launch");
            ctx.stream.synchronize().expect("portable sync");
            let expected = f32_bits(&ctx, &portable, shape.m * shape.n);
            for tile in specialized {
                let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                    .expect("SM120 output");
                let operands = InferenceFwdOperands {
                    c: typed(&output, WeightDtype::F32),
                    ..portable_operands
                };
                inference_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                ctx.stream.synchronize().expect("first SM120 TF32 sync");
                let first = f32_bits(&ctx, &output, shape.m * shape.n);
                inference_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("repeated {tile:?}: {error}"));
                ctx.stream.synchronize().expect("second SM120 TF32 sync");
                assert_eq!(
                    first,
                    f32_bits(&ctx, &output, shape.m * shape.n),
                    "SM120 repeat drift for {shape:?} {tile:?}"
                );
                assert_eq!(
                    first,
                    expected,
                    "SM120/portable drift for {shape:?} {tile:?} bias={}",
                    bias_ptr.is_some()
                );
            }
            let selected = inference_forward(
                &ctx,
                portable_operands.c,
                portable_operands.x,
                portable_operands.w,
                bias_ptr,
                (shape.m, shape.k, shape.n),
            )
            .expect("automatic SM120 TF32 launch");
            assert_eq!(selected, InferenceTile::Tf32Sm120M64S2);
        }
    }
}

#[test]
#[ignore = "requires a quiet RTX 5090 and screens the SM120 TF32 producer-warp candidate"]
fn fixed_sm120_tf32_producer_warp_candidate_screen() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert_eq!(
        ctx.stream
            .context()
            .compute_capability()
            .expect("CUDA compute capability"),
        (12, 0),
        "producer-warp candidate screen is qualified on SM120",
    );
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32);

    let incumbent = InferenceTile::Tf32Sm120M64S2;
    let candidate = InferenceTile::Tf32Sm120M64S2ProducerWarp;
    for (label, shape) in [
        (
            "A",
            InferenceShape {
                m: 4621,
                k: 384,
                n: 1928,
            },
        ),
        (
            "B",
            InferenceShape {
                m: 4621,
                k: 768,
                n: 2304,
            },
        ),
        (
            "D",
            InferenceShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
        ),
    ] {
        let output_len = shape.m * shape.n;
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("B allocation");
        let bias =
            DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
        let incumbent_output =
            DtypedBuf::zeros(&ctx.stream, output_len, WeightDtype::F32).expect("incumbent output");
        let candidate_output =
            DtypedBuf::zeros(&ctx.stream, output_len, WeightDtype::F32).expect("candidate output");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x6496_a000))
            .expect("A upload");
        b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x6496_b000))
            .expect("B upload");
        bias.upload_f32(&ctx.stream, &synth(shape.n, 0x6496_c000))
            .expect("bias upload");

        for (bias_label, bias_ptr) in [("none", None), ("bias", Some(bias.cached_ptr()))] {
            let incumbent_operands = InferenceFwdOperands {
                c: typed(&incumbent_output, WeightDtype::F32),
                x: typed(&a, WeightDtype::F32),
                w: typed(&b, WeightDtype::F32),
                bias_ptr,
            };
            let candidate_operands = InferenceFwdOperands {
                c: typed(&candidate_output, WeightDtype::F32),
                ..incumbent_operands
            };
            inference_forward_with_tile(&ctx, incumbent_operands, shape, incumbent)
                .expect("incumbent exact-bit launch");
            inference_forward_with_tile(&ctx, candidate_operands, shape, candidate)
                .expect("candidate exact-bit launch");
            ctx.stream.synchronize().expect("exact-bit synchronization");
            let expected = f32_bits(&ctx, &incumbent_output, output_len);
            assert_eq!(
                f32_bits(&ctx, &candidate_output, output_len),
                expected,
                "candidate bit drift for {label}/{bias_label}"
            );

            let graph = unsafe {
                capture_into_graph(&ctx.stream, || {
                    inference_forward_with_tile(&ctx, candidate_operands, shape, candidate)
                })
            }
            .expect("capture producer-warp graph");
            assert_eq!(
                single_graph_kernel_name(&graph, "producer-warp candidate"),
                "nn_sm120_tma_tf32_m64n64_bk32_s2_producer_warp"
            );
            for replay in 0..10 {
                graph.launch().expect("producer-warp graph launch");
                ctx.stream.synchronize().expect("producer-warp graph sync");
                assert_eq!(
                    f32_bits(&ctx, &candidate_output, output_len),
                    expected,
                    "candidate graph drift for {label}/{bias_label} replay {replay}"
                );
            }

            for _ in 0..128 {
                inference_forward_with_tile(&ctx, incumbent_operands, shape, incumbent)
                    .expect("incumbent warmup");
                inference_forward_with_tile(&ctx, candidate_operands, shape, candidate)
                    .expect("candidate warmup");
            }
            ctx.stream.synchronize().expect("candidate warmup sync");
            let incumbent_iterations =
                fixed_tile_window_iterations(&ctx, incumbent_operands, shape, incumbent);
            let candidate_iterations =
                fixed_tile_window_iterations(&ctx, candidate_operands, shape, candidate);
            for (order, candidate_first) in [
                ("candidate_then_incumbent", true),
                ("incumbent_then_candidate", false),
            ] {
                let mut ratios = Vec::with_capacity(101);
                for _ in 0..101 {
                    let (candidate_us, incumbent_us) = if candidate_first {
                        (
                            fixed_tile_window_us(
                                &ctx,
                                candidate_operands,
                                shape,
                                candidate,
                                candidate_iterations,
                            ),
                            fixed_tile_window_us(
                                &ctx,
                                incumbent_operands,
                                shape,
                                incumbent,
                                incumbent_iterations,
                            ),
                        )
                    } else {
                        let incumbent_us = fixed_tile_window_us(
                            &ctx,
                            incumbent_operands,
                            shape,
                            incumbent,
                            incumbent_iterations,
                        );
                        let candidate_us = fixed_tile_window_us(
                            &ctx,
                            candidate_operands,
                            shape,
                            candidate,
                            candidate_iterations,
                        );
                        (candidate_us, incumbent_us)
                    };
                    ratios.push(candidate_us / incumbent_us);
                }
                ratios.sort_by(f64::total_cmp);
                println!(
                    "TF32 producer-warp label={label} bias={bias_label} order={order} candidate_over_incumbent_p50={:.9} p95={:.9}",
                    percentile(&ratios, 0.50),
                    percentile(&ratios, 0.95),
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_tf32_graph_replay_is_bit_exact_and_cold_capture_fails() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    if !matches!(
        ctx.stream
            .context()
            .compute_capability()
            .expect("CUDA compute capability"),
        (12, 0) | (12, 1)
    ) {
        return;
    }
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32);
    let shape = InferenceShape {
        m: 65,
        k: 36,
        n: 68,
    };
    let a =
        DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32).expect("A allocation");
    let b =
        DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32).expect("B allocation");
    let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
        .expect("output allocation");
    a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xc120a))
        .expect("A upload");
    b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xc120b))
        .expect("B upload");
    let run = || {
        inference_forward(
            &ctx,
            typed(&output, WeightDtype::F32),
            typed(&a, WeightDtype::F32),
            typed(&b, WeightDtype::F32),
            None,
            (shape.m, shape.k, shape.n),
        )
        .and_then(|tile| {
            if tile == InferenceTile::Tf32Sm120M64S2 {
                Ok(())
            } else {
                Err(format!("unexpected production tile {tile:?}"))
            }
        })
    };
    run().expect("warm tensor maps");
    ctx.stream.synchronize().expect("eager synchronization");
    let eager = f32_bits(&ctx, &output, shape.m * shape.n);
    let graph = unsafe { capture_into_graph(&ctx.stream, run) }.expect("capture Fixed SM120 TF32");
    for replay in 0..10 {
        graph.launch().expect("launch Fixed SM120 graph");
        ctx.stream.synchronize().expect("graph synchronization");
        assert_eq!(
            f32_bits(&ctx, &output, shape.m * shape.n),
            eager,
            "Fixed SM120 graph replay {replay} changed bits"
        );
    }

    let cold_a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
        .expect("cold A allocation");
    let cold_b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
        .expect("cold B allocation");
    let cold_capture = unsafe {
        capture_into_graph(&ctx.stream, || {
            inference_forward(
                &ctx,
                typed(&output, WeightDtype::F32),
                typed(&cold_a, WeightDtype::F32),
                typed(&cold_b, WeightDtype::F32),
                None,
                (shape.m, shape.k, shape.n),
            )
            .map(|_| ())
        })
    };
    let error = match cold_capture {
        Ok(_) => panic!("cold tensor maps must not be encoded during capture"),
        Err(error) => error,
    };
    assert!(
        error.contains("must be prepared before graph capture"),
        "unexpected cold-capture error: {error}"
    );
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn fixed_tf32_forward_is_batch_prefix_invariant() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32);
    let (large_m, k, n) = (129usize, 96usize, 196usize);
    let large_a_host = synth(large_m * k, 0xba7c4);
    let b_host = synth(k * n, 0xb32);
    let bias_host = synth(n, 0xb1a5);
    let small_a = DtypedBuf::zeros(&ctx.stream, k, WeightDtype::F32).expect("small A");
    let large_a = DtypedBuf::zeros(&ctx.stream, large_m * k, WeightDtype::F32).expect("large A");
    let b = DtypedBuf::zeros(&ctx.stream, k * n, WeightDtype::F32).expect("B");
    let bias = DtypedBuf::zeros(&ctx.stream, n, WeightDtype::F32).expect("bias");
    let small_c = DtypedBuf::zeros(&ctx.stream, n, WeightDtype::F32).expect("small C");
    let large_c = DtypedBuf::zeros(&ctx.stream, large_m * n, WeightDtype::F32).expect("large C");
    small_a
        .upload_f32(&ctx.stream, &large_a_host[..k])
        .expect("small A upload");
    large_a
        .upload_f32(&ctx.stream, &large_a_host)
        .expect("large A upload");
    b.upload_f32(&ctx.stream, &b_host).expect("B upload");
    bias.upload_f32(&ctx.stream, &bias_host)
        .expect("bias upload");
    let small_tile = inference_forward(
        &ctx,
        typed(&small_c, WeightDtype::F32),
        typed(&small_a, WeightDtype::F32),
        typed(&b, WeightDtype::F32),
        Some(bias.cached_ptr()),
        (1, k, n),
    )
    .expect("small TF32 launch");
    let large_tile = inference_forward(
        &ctx,
        typed(&large_c, WeightDtype::F32),
        typed(&large_a, WeightDtype::F32),
        typed(&b, WeightDtype::F32),
        Some(bias.cached_ptr()),
        (large_m, k, n),
    )
    .expect("large TF32 launch");
    ctx.stream.synchronize().expect("TF32 prefix sync");
    assert_ne!(small_tile, large_tile, "test must cross selector rungs");
    assert_eq!(
        f32_bits(&ctx, &small_c, n),
        f32_bits(&ctx, &large_c, large_m * n)[..n],
        "TF32 first-row bits changed with batch size"
    );
}

#[test]
#[ignore = "requires exclusive CC8.9/142-SM CUDA13.2 Ada device"]
fn fixed_sm89_tf32_c_auto_prefix_special_bias_graph_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9));
    assert_eq!(device.multiprocessor_count(), 142);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let compiler = ctx.kernels.compiler_identity();
    assert_eq!(compiler.nvrtc_version, (13, 2));
    assert!(compiler.nvrtc_library_known);
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32);

    let shape = InferenceShape {
        m: 4621,
        k: 1928,
        n: 384,
    };
    let a_host = synth(shape.m * shape.k, 0xc089_a001);
    let b_host = synth(shape.k * shape.n, 0xc089_b001);
    let special_bias_bits = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x8080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffc5_4321,
        0x3a80_0000,
    ];
    let bias_host = (0..shape.n)
        .map(|index| f32::from_bits(special_bias_bits[index % special_bias_bits.len()]))
        .collect::<Vec<_>>();
    let a = DtypedBuf::zeros(&ctx.stream, a_host.len(), WeightDtype::F32).expect("C-row A");
    let b = DtypedBuf::zeros(&ctx.stream, b_host.len(), WeightDtype::F32).expect("C-row B");
    let bias =
        DtypedBuf::zeros(&ctx.stream, bias_host.len(), WeightDtype::F32).expect("C-row bias");
    a.upload_f32(&ctx.stream, &a_host).expect("C-row A upload");
    b.upload_f32(&ctx.stream, &b_host).expect("C-row B upload");
    bias.upload_f32(&ctx.stream, &bias_host)
        .expect("C-row bias upload");

    for bias_ptr in [None, Some(bias.cached_ptr())] {
        let auto = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("C-row AUTO output");
        let old_auto = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("C-row old AUTO output");
        let prefix =
            DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("C-row prefix output");
        let inputs = InferenceFwdOperands {
            c: typed(&auto, WeightDtype::F32),
            x: typed(&a, WeightDtype::F32),
            w: typed(&b, WeightDtype::F32),
            bias_ptr,
        };
        inference_forward_with_tile(
            &ctx,
            InferenceFwdOperands {
                c: typed(&old_auto, WeightDtype::F32),
                ..inputs
            },
            shape,
            InferenceTile::Tf32M128S2,
        )
        .expect("forced prior C AUTO");
        let launch_auto = || {
            inference_forward(
                &ctx,
                inputs.c,
                inputs.x,
                inputs.w,
                bias_ptr,
                (shape.m, shape.k, shape.n),
            )
            .and_then(|tile| {
                (tile == InferenceTile::Tf32RnaM128N128S3)
                    .then_some(())
                    .ok_or_else(|| format!("unexpected Ada C AUTO tile {tile:?}"))
            })
        };
        launch_auto().expect("eager Ada C AUTO");
        let prefix_tile = inference_forward(
            &ctx,
            typed(&prefix, WeightDtype::F32),
            inputs.x,
            inputs.w,
            bias_ptr,
            (1, shape.k, shape.n),
        )
        .expect("Ada C prefix AUTO");
        assert_eq!(prefix_tile, InferenceTile::Tf32M16S4);
        ctx.stream.synchronize().expect("Ada C eager sync");
        let eager = f32_bits(&ctx, &auto, shape.m * shape.n);
        assert_eq!(eager, f32_bits(&ctx, &old_auto, shape.m * shape.n));
        assert_eq!(f32_bits(&ctx, &prefix, shape.n), eager[..shape.n]);

        let graph = unsafe { capture_into_graph(&ctx.stream, launch_auto) }
            .expect("capture Ada C AUTO graph");
        assert_eq!(
            single_graph_kernel_name(&graph, "Ada C AUTO"),
            "nn_rna_wide_tf32_m128n128_bk32_s3",
        );
        graph.launch().expect("replay Ada C AUTO graph");
        ctx.stream.synchronize().expect("Ada C graph sync");
        assert_eq!(f32_bits(&ctx, &auto, shape.m * shape.n), eager);
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits TF32 performance data"]
fn fixed_tf32_hot_shapes_smoke() {
    let shapes = [
        InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let mut tiles = vec![
        InferenceTile::Tf32M128S2,
        InferenceTile::Tf32M128S3,
        InferenceTile::Tf32M64S2,
        InferenceTile::Tf32M64S3,
        InferenceTile::Tf32M16S4,
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    if matches!(
        ctx.stream
            .context()
            .compute_capability()
            .expect("CUDA compute capability"),
        (12, 0) | (12, 1)
    ) {
        tiles.extend([
            InferenceTile::Tf32Sm120M128S2,
            InferenceTile::Tf32Sm120M128S3,
            InferenceTile::Tf32Sm120M64N128S2,
            InferenceTile::Tf32Sm120M64N128S3,
            InferenceTile::Tf32Sm120M64S2,
        ]);
    }
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32);
    for shape in shapes {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("B allocation");
        let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("C allocation");
        let operands = InferenceFwdOperands {
            c: typed(&c, WeightDtype::F32),
            x: typed(&a, WeightDtype::F32),
            w: typed(&b, WeightDtype::F32),
            bias_ptr: None,
        };
        for &tile in &tiles {
            let elapsed_us = average_us(&ctx, || {
                inference_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
            });
            println!(
                "m={} k={} n={} tile={tile:?} elapsed_us={elapsed_us:.3}",
                shape.m, shape.k, shape.n
            );
        }
        let mut selected = InferenceTile::Legacy;
        let auto_us = average_us(&ctx, || {
            selected = inference_forward(
                &ctx,
                operands.c,
                operands.x,
                operands.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("automatic Fixed TF32 launch");
        });
        ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
        let cublas_us = average_us(&ctx, || {
            gpu_gemm_typed_forward_raw(
                &ctx,
                operands.c,
                operands.x,
                operands.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("cuBLAS TF32 launch");
        });
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        println!(
            "m={} k={} n={} auto={selected:?} auto_us={auto_us:.3} cublas_tf32_us={cublas_us:.3} auto_over_cublas={:.5}",
            shape.m,
            shape.k,
            shape.n,
            auto_us / cublas_us
        );
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device and emits specialized TF32 reference data"]
fn triad_sm120_tf32_nn_reference_smoke() {
    let shapes = [
        InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0), "SM120 required");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    for shape in shapes {
        for spec in tf32_route_specs(ModuleKind::TriadSm120)
            .iter()
            .filter(|spec| spec.op == ResolvedGemmOp::Nn)
        {
            let request = PhysicalQualificationRequest::contiguous(
                ResolvedGemmOp::Nn,
                (shape.m, shape.k, shape.n),
                PhysicalQualificationRoute::Tf32Forced(spec.route),
            );
            let mut qualified = qualify_physical_launch(&ctx, request)
                .unwrap_or_else(|error| panic!("qualify {}: {error}", spec.symbol));
            let iterations = 200;
            let elapsed_us = qualified
                .measure_eager_window_ms(&ctx, iterations)
                .unwrap_or_else(|error| panic!("measure {}: {error}", spec.symbol))
                * 1000.0
                / iterations as f64;
            println!(
                "m={} k={} n={} specialized_symbol={} elapsed_us={elapsed_us:.3}",
                shape.m, shape.k, shape.n, spec.symbol
            );
        }
    }
}

/// One body, two families: the Fixed inference family and the Triad
/// family carry kernels of the same tile geometry. This survey times each
/// pair on the same NN shapes so the slower body can be retired on evidence.
#[test]
#[ignore = "requires an otherwise idle SM120 CUDA device and emits the Inference/Triad pairwise survey"]
fn fixed_vs_triad_pairwise_census() {
    fixed_sm120_tf32_bd_environment_preflight("pairwise survey")
        .expect("pairwise survey preflight");
    let shapes = [
        InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        InferenceShape {
            m: 2048,
            k: 768,
            n: 3072,
        },
        InferenceShape {
            m: 2048,
            k: 1536,
            n: 768,
        },
        InferenceShape {
            m: 10400,
            k: 768,
            n: 384,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let sm120 = matches!(device.compute_capability, (12, 0) | (12, 1));
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32);
    let iterations = 200;
    // The half routes stage an upcast scratch that cannot grow once a
    // qualification has captured a graph: size it for every request first.
    let half_requests = shapes
        .iter()
        .flat_map(|shape| {
            [WeightDtype::Bf16, WeightDtype::F16]
                .into_iter()
                .flat_map(move |dtype| {
                    [TcTile::Tile64, TcTile::Thin16]
                        .into_iter()
                        .map(move |tile| {
                            PhysicalQualificationRequest::contiguous(
                                ResolvedGemmOp::Nn,
                                (shape.m, shape.k, shape.n),
                                PhysicalQualificationRoute::HalfForced { dtype, tile },
                            )
                        })
                })
        })
        .collect::<Vec<_>>();
    presize_physical_qualification_suite(&ctx, &half_requests)
        .expect("pre-size the half qualification scratch");
    let triad_us = |request: PhysicalQualificationRequest, label: &str| -> Option<f64> {
        match qualify_physical_launch(&ctx, request) {
            Ok(mut qualified) => Some(
                qualified
                    .measure_eager_window_ms(&ctx, iterations)
                    .unwrap_or_else(|error| panic!("measure {label}: {error}"))
                    * 1000.0
                    / iterations as f64,
            ),
            Err(error) => {
                println!("triad {label}: unavailable ({error})");
                None
            }
        }
    };
    let fixed_us = |operands: InferenceFwdOperands,
                    shape: InferenceShape,
                    tile: InferenceTile|
     -> Option<f64> {
        if let Err(error) = inference_forward_with_tile(&ctx, operands, shape, tile) {
            println!("fixed {tile:?}: unavailable ({error})");
            return None;
        }
        Some(average_us(&ctx, || {
            inference_forward_with_tile(&ctx, operands, shape, tile)
                .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
        }))
    };
    let report = |shape: InferenceShape, pair: &str, fixed: Option<f64>, triad: Option<f64>| {
        let verdict = match (fixed, triad) {
            (Some(fixed), Some(triad)) if fixed < triad => {
                format!("fixed faster by {:.3}x", triad / fixed)
            }
            (Some(fixed), Some(triad)) => format!("triad faster by {:.3}x", fixed / triad),
            _ => "one side unavailable".to_string(),
        };
        let show =
            |value: Option<f64>| value.map_or("n/a".to_string(), |value| format!("{value:.3}"));
        println!(
            "m={} k={} n={} pair={pair} fixed_us={} triad_us={} verdict={verdict}",
            shape.m,
            shape.k,
            shape.n,
            show(fixed),
            show(triad)
        );
    };
    for shape in shapes {
        let dims = (shape.m, shape.k, shape.n);
        // TF32: the five portable bodies and, on SM120, the five TMA bodies.
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32).expect("A");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32).expect("B");
        let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32).expect("C");
        let f32_operands = InferenceFwdOperands {
            c: typed(&c, WeightDtype::F32),
            x: typed(&a, WeightDtype::F32),
            w: typed(&b, WeightDtype::F32),
            bias_ptr: None,
        };
        let portable = [
            (
                "tf32_m128n64_s2",
                InferenceTile::Tf32M128S2,
                Tf32PortableTile::M128N64,
                Tf32PortableStages::S2,
            ),
            (
                "tf32_m128n64_s3",
                InferenceTile::Tf32M128S3,
                Tf32PortableTile::M128N64,
                Tf32PortableStages::S3,
            ),
            (
                "tf32_m64n64_s2",
                InferenceTile::Tf32M64S2,
                Tf32PortableTile::M64N64,
                Tf32PortableStages::S2,
            ),
            (
                "tf32_m64n64_s3",
                InferenceTile::Tf32M64S3,
                Tf32PortableTile::M64N64,
                Tf32PortableStages::S3,
            ),
            (
                "tf32_m16n32_s4",
                InferenceTile::Tf32M16S4,
                Tf32PortableTile::M16N32,
                Tf32PortableStages::S4,
            ),
        ];
        for (pair, fixed_tile, tile, stages) in portable {
            let route = Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute { tile, stages });
            let request = PhysicalQualificationRequest::contiguous(
                ResolvedGemmOp::Nn,
                dims,
                PhysicalQualificationRoute::Tf32Forced(route),
            );
            report(
                shape,
                pair,
                fixed_us(f32_operands, shape, fixed_tile),
                triad_us(request, pair),
            );
        }
        if sm120 {
            let tma = [
                (
                    "tf32_sm120_m128n64_s2",
                    InferenceTile::Tf32Sm120M128S2,
                    Tf32Sm120Tile::M128N64,
                    Tf32Sm120Stages::S2,
                ),
                (
                    "tf32_sm120_m128n64_s3",
                    InferenceTile::Tf32Sm120M128S3,
                    Tf32Sm120Tile::M128N64,
                    Tf32Sm120Stages::S3,
                ),
                (
                    "tf32_sm120_m64n128_s2",
                    InferenceTile::Tf32Sm120M64N128S2,
                    Tf32Sm120Tile::M64N128,
                    Tf32Sm120Stages::S2,
                ),
                (
                    "tf32_sm120_m64n128_s3",
                    InferenceTile::Tf32Sm120M64N128S3,
                    Tf32Sm120Tile::M64N128,
                    Tf32Sm120Stages::S3,
                ),
                (
                    "tf32_sm120_m64n64_s2",
                    InferenceTile::Tf32Sm120M64S2,
                    Tf32Sm120Tile::M64N64,
                    Tf32Sm120Stages::S2,
                ),
            ];
            for (pair, fixed_tile, tile, stages) in tma {
                let route = Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route { tile, stages });
                let request = PhysicalQualificationRequest::contiguous(
                    ResolvedGemmOp::Nn,
                    dims,
                    PhysicalQualificationRoute::Tf32Forced(route),
                );
                report(
                    shape,
                    pair,
                    fixed_us(f32_operands, shape, fixed_tile),
                    triad_us(request, pair),
                );
            }
        }
        // Half: the portable 64x64 and 16x32 bodies, and on SM120 the five TMA tiles.
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C");
            let half_operands = InferenceFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            for (pair, fixed_tile, tile) in [
                ("half_tc64", InferenceTile::Tc64, TcTile::Tile64),
                ("half_tc16", InferenceTile::Tc16, TcTile::Thin16),
            ] {
                let pair = format!("{pair}_{dtype:?}");
                let request = PhysicalQualificationRequest::contiguous(
                    ResolvedGemmOp::Nn,
                    dims,
                    PhysicalQualificationRoute::HalfForced { dtype, tile },
                );
                report(
                    shape,
                    &pair,
                    fixed_us(half_operands, shape, fixed_tile),
                    triad_us(request, &pair),
                );
            }
            if !sm120 {
                continue;
            }
            let caps = ctx.kernels.sm120_device_caps().expect("SM120 device caps");
            let target = ctx.kernels.sm120_target_candidate().expect("SM120 target");
            let sm120_shape = Sm120Shape::contiguous(Sm120Op::Nn, dims);
            let tiles = [
                (
                    "half_sm120_m64n64_bk64_s2",
                    InferenceSm120HalfTile::M64N64Bk64S2,
                    Sm120Tile::M64N64,
                    Sm120Bk::Bk64,
                    Sm120Stages::S2,
                ),
                (
                    "half_sm120_m64n128_bk64_s2",
                    InferenceSm120HalfTile::M64N128Bk64S2,
                    Sm120Tile::M64N128,
                    Sm120Bk::Bk64,
                    Sm120Stages::S2,
                ),
                (
                    "half_sm120_m128n64_bk32_s3",
                    InferenceSm120HalfTile::M128N64Bk32S3,
                    Sm120Tile::M128N64,
                    Sm120Bk::Bk32,
                    Sm120Stages::S3,
                ),
                (
                    "half_sm120_m128n128_bk32_s2",
                    InferenceSm120HalfTile::M128N128Bk32S2,
                    Sm120Tile::M128N128,
                    Sm120Bk::Bk32,
                    Sm120Stages::S2,
                ),
                (
                    "half_sm120_m128n128_bk32_s3",
                    InferenceSm120HalfTile::M128N128Bk32S3,
                    Sm120Tile::M128N128,
                    Sm120Bk::Bk32,
                    Sm120Stages::S3,
                ),
            ];
            for (pair, fixed_tile, tile, bk, stages) in tiles {
                let pair = format!("{pair}_{dtype:?}");
                let physical = Sm120PhysicalRoute {
                    tile,
                    bk,
                    stages,
                    schedule: Sm120Schedule::Tiled,
                };
                let triad = (|| -> Result<f64, String> {
                    let route = resolve_sm120_forced(
                        caps,
                        Some(target),
                        Sm120ForcedRoute {
                            op: Sm120Op::Nn,
                            dtype,
                            physical,
                            shape: sm120_shape,
                        },
                    )?
                    .ok_or_else(|| "route declined".to_string())?;
                    let maps = prepare_sm120_tensor_maps(
                        &ctx.stream,
                        &ctx.kernels,
                        Sm120MapRequest {
                            op: Sm120Op::Nn,
                            dtype,
                            tile,
                            bk,
                            a_ptr: half_operands.x.ptr,
                            b_ptr: half_operands.w.ptr,
                            shape: sm120_shape,
                        },
                    )?;
                    let prepared = prepare_sm120_tma_forced(
                        &ctx.stream,
                        &ctx.kernels,
                        route,
                        &maps,
                        Sm120LaunchOperands {
                            output_ptr: half_operands.c.ptr,
                            bias_ptr: 0,
                            alpha: 1.0,
                            beta: 0.0,
                        },
                    )?;
                    for _ in 0..WARMUPS {
                        launch_sm120_tma_prepared(&ctx.stream, &ctx.kernels, &prepared)?;
                    }
                    let start = ctx
                        .stream
                        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .map_err(|error| format!("record start: {error:?}"))?;
                    for _ in 0..iterations {
                        launch_sm120_tma_prepared(&ctx.stream, &ctx.kernels, &prepared)?;
                    }
                    let end = ctx
                        .stream
                        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .map_err(|error| format!("record end: {error:?}"))?;
                    Ok(f64::from(
                        start
                            .elapsed_ms(&end)
                            .map_err(|error| format!("measure: {error:?}"))?,
                    ) * 1000.0
                        / iterations as f64)
                })();
                let triad = match triad {
                    Ok(value) => Some(value),
                    Err(error) => {
                        println!("triad {pair}: unavailable ({error})");
                        None
                    }
                };
                report(
                    shape,
                    &pair,
                    fixed_us(half_operands, shape, InferenceTile::Sm120Half(fixed_tile)),
                    triad,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn half_to_f32_portable_ladder_matches_legacy_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in [
            InferenceShape { m: 9, k: 33, n: 17 },
            InferenceShape {
                m: 65,
                k: 63,
                n: 127,
            },
            InferenceShape {
                m: 128,
                k: 64,
                n: 128,
            },
            InferenceShape {
                m: 4621,
                k: 384,
                n: 384,
            },
        ] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let baseline = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("baseline allocation");
            let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("candidate allocation");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa11ce))
                .expect("A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb0b))
                .expect("B upload");
            let baseline_operands = InferenceFwdOperands {
                c: typed(&baseline, WeightDtype::F32),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let candidate_operands = InferenceFwdOperands {
                c: typed(&candidate, WeightDtype::F32),
                ..baseline_operands
            };
            inference_forward_with_tile(&ctx, baseline_operands, shape, InferenceTile::Legacy)
                .expect("legacy mixed-output launch");
            let baseline_bits = f32_bits(&ctx, &baseline, shape.m * shape.n);
            for tile in [
                InferenceTile::Tc16,
                InferenceTile::Tc64,
                InferenceTile::Tc128,
            ] {
                inference_forward_with_tile(&ctx, candidate_operands, shape, tile)
                    .unwrap_or_else(|error| panic!("mixed-output {tile:?} launch: {error}"));
                ctx.stream.synchronize().expect("mixed-output sync");
                assert_eq!(
                    f32_bits(&ctx, &candidate, shape.m * shape.n),
                    baseline_bits,
                    "{dtype:?} to F32 {tile:?} changed bits for M{} K{} N{}",
                    shape.m,
                    shape.k,
                    shape.n,
                );
            }
            let selected = inference_forward(
                &ctx,
                candidate_operands.c,
                candidate_operands.x,
                candidate_operands.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("production mixed-output launch");
            assert!(
                matches!(
                    selected,
                    InferenceTile::Tc16
                        | InferenceTile::Tc64
                        | InferenceTile::Tc128
                        | InferenceTile::Sm120Half(_)
                ),
                "production mixed-output route retained {selected:?}"
            );
            ctx.stream
                .synchronize()
                .expect("production mixed-output sync");
            assert_eq!(
                f32_bits(&ctx, &candidate, shape.m * shape.n),
                baseline_bits,
                "{dtype:?} to F32 production changed bits for M{} K{} N{}",
                shape.m,
                shape.k,
                shape.n,
            );
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn half_to_f32_sm120_tma_ladder_matches_portable_bits() {
    const GUARD: usize = 8;
    const SENTINEL: f32 = 19.25;
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(
        matches!(device.compute_capability, (12, 0) | (12, 1)),
        "SM120 or SM121 required"
    );
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for (tile, shape) in [
        (
            InferenceSm120HalfTile::M64N64Bk64S2,
            InferenceShape {
                m: 65,
                k: 136,
                n: 72,
            },
        ),
        (
            InferenceSm120HalfTile::M64N128Bk64S2,
            InferenceShape {
                m: 65,
                k: 136,
                n: 136,
            },
        ),
        (
            InferenceSm120HalfTile::M128N64Bk32S3,
            InferenceShape {
                m: 129,
                k: 104,
                n: 72,
            },
        ),
        (
            InferenceSm120HalfTile::M128N128Bk32S2,
            InferenceShape {
                m: 129,
                k: 72,
                n: 136,
            },
        ),
    ] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let bias =
                DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x120a))
                .expect("A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x120b))
                .expect("B upload");
            bias.upload_f32(&ctx.stream, &synth(shape.n, 0x120c))
                .expect("bias upload");

            for bias_ptr in [None, Some(bias.cached_ptr())] {
                let output_len = shape.m * shape.n;
                let portable = DtypedBuf::zeros(&ctx.stream, output_len, WeightDtype::F32)
                    .expect("portable allocation");
                let portable_operands = InferenceFwdOperands {
                    c: typed(&portable, WeightDtype::F32),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr,
                };
                inference_forward_with_tile(&ctx, portable_operands, shape, InferenceTile::Tc128)
                    .expect("portable mixed-output launch");
                ctx.stream.synchronize().expect("portable sync");
                let expected = f32_bits(&ctx, &portable, output_len);

                for output_offset in [0usize, 1] {
                    let output_start = GUARD + output_offset;
                    let storage_len = output_start + output_len + GUARD;
                    let candidate = DtypedBuf::zeros(&ctx.stream, storage_len, WeightDtype::F32)
                        .expect("candidate allocation");
                    let candidate_operands = InferenceFwdOperands {
                        c: TypedPtr {
                            ptr: candidate.cached_ptr()
                                + (output_start * WeightDtype::F32.size_bytes()) as u64,
                            dtype: WeightDtype::F32,
                        },
                        ..portable_operands
                    };
                    for poison in [-7.0f32, 9.0] {
                        let mut reset = vec![SENTINEL; storage_len];
                        reset[output_start..output_start + output_len].fill(poison);
                        candidate
                            .upload_f32(&ctx.stream, &reset)
                            .expect("candidate reset");
                        inference_forward_with_tile(
                            &ctx,
                            candidate_operands,
                            shape,
                            InferenceTile::Sm120Half(tile),
                        )
                        .unwrap_or_else(|error| {
                            panic!("SM120 mixed-output {dtype:?} {tile:?}: {error}")
                        });
                        ctx.stream.synchronize().expect("SM120 mixed-output sync");
                        let observed = f32_bits(&ctx, &candidate, storage_len);
                        assert!(
                            observed[..output_start]
                                .iter()
                                .all(|&bits| bits == SENTINEL.to_bits())
                        );
                        assert!(
                            observed[output_start + output_len..]
                                .iter()
                                .all(|&bits| bits == SENTINEL.to_bits())
                        );
                        assert_eq!(
                            observed[output_start..output_start + output_len],
                            expected,
                            "{dtype:?} SM120 mixed-output {tile:?} changed portable bits; bias={} offset={output_offset}",
                            bias_ptr.is_some(),
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn half_to_f32_sm120_graph_replay_is_bit_exact_and_cold_capture_fails() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    let dtype = WeightDtype::Bf16;
    let shape = InferenceShape {
        m: 128,
        k: 96,
        n: 1536,
    };
    let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
    let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
    let reference = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
        .expect("reference allocation");
    let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
        .expect("output allocation");
    a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xc120a))
        .expect("A upload");
    b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xc120b))
        .expect("B upload");
    let base_operands = InferenceFwdOperands {
        c: typed(&reference, WeightDtype::F32),
        x: typed(&a, dtype),
        w: typed(&b, dtype),
        bias_ptr: None,
    };
    inference_forward_with_tile(&ctx, base_operands, shape, InferenceTile::Tc128)
        .expect("portable reference launch");
    ctx.stream.synchronize().expect("portable reference sync");
    let expected = f32_bits(&ctx, &reference, shape.m * shape.n);

    let run = || {
        inference_forward(
            &ctx,
            typed(&output, WeightDtype::F32),
            base_operands.x,
            base_operands.w,
            None,
            (shape.m, shape.k, shape.n),
        )
        .and_then(|tile| match tile {
            InferenceTile::Sm120Half(_) => Ok(()),
            _ => Err(format!("unexpected production tile {tile:?}")),
        })
    };
    run().expect("warm mixed-output tensor maps");
    ctx.stream.synchronize().expect("eager synchronization");
    assert_eq!(f32_bits(&ctx, &output, shape.m * shape.n), expected);
    let graph =
        unsafe { capture_into_graph(&ctx.stream, run) }.expect("capture Fixed SM120 mixed-output");
    assert!(
        single_graph_kernel_name(&graph, "SM120 mixed-output").contains("_f32out_bf16"),
        "captured graph must retain the physical F32-output kernel"
    );
    for replay in 0..10 {
        graph
            .launch()
            .expect("launch Fixed SM120 mixed-output graph");
        ctx.stream.synchronize().expect("graph synchronization");
        assert_eq!(
            f32_bits(&ctx, &output, shape.m * shape.n),
            expected,
            "Fixed SM120 mixed-output graph replay {replay} changed bits"
        );
    }

    let cold_a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("cold A");
    let cold_b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("cold B");
    let cold_capture = unsafe {
        capture_into_graph(&ctx.stream, || {
            inference_forward(
                &ctx,
                typed(&output, WeightDtype::F32),
                typed(&cold_a, dtype),
                typed(&cold_b, dtype),
                None,
                (shape.m, shape.k, shape.n),
            )
            .map(|_| ())
        })
    };
    let error = match cold_capture {
        Ok(_) => panic!("cold mixed-output tensor maps must not be encoded during capture"),
        Err(error) => error,
    };
    assert!(
        error.contains("must be prepared before graph capture"),
        "unexpected cold-capture error: {error}"
    );
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn half_sm120_exceptional_values_match_portable_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let shape = InferenceShape { m: 65, k: 8, n: 8 };
    let exceptional = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0x7f81_2345),
        65_504.0,
        -65_504.0,
        2.0f32.powi(-14),
        -2.0f32.powi(-14),
        1.0,
        -1.0,
    ];
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let a_host = (0..shape.m * shape.k)
            .map(|index| exceptional[index % exceptional.len()])
            .collect::<Vec<_>>();
        let b_host = (0..shape.k * shape.n)
            .map(|index| exceptional[(index * 5 + 3) % exceptional.len()])
            .collect::<Vec<_>>();
        let bias_host = (0..shape.n)
            .map(|index| exceptional[(index * 7 + 4) % exceptional.len()])
            .collect::<Vec<_>>();
        let a = DtypedBuf::zeros(&ctx.stream, a_host.len(), dtype).expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, b_host.len(), dtype).expect("B allocation");
        let bias = DtypedBuf::zeros(&ctx.stream, bias_host.len(), WeightDtype::F32)
            .expect("bias allocation");
        a.upload_f32(&ctx.stream, &a_host).expect("A upload");
        b.upload_f32(&ctx.stream, &b_host).expect("B upload");
        bias.upload_f32(&ctx.stream, &bias_host)
            .expect("bias upload");
        for output_dtype in [dtype, WeightDtype::F32] {
            for bias_ptr in [None, Some(bias.cached_ptr())] {
                let reference = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, output_dtype)
                    .expect("reference allocation");
                let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, output_dtype)
                    .expect("candidate allocation");
                let reference_operands = InferenceFwdOperands {
                    c: typed(&reference, output_dtype),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr,
                };
                inference_forward_with_tile(&ctx, reference_operands, shape, InferenceTile::Tc128)
                    .expect("portable exceptional reference");
                ctx.stream.synchronize().expect("portable exceptional sync");
                let expected = f32_bits(&ctx, &reference, shape.m * shape.n);
                for tile in InferenceSm120HalfTile::ALL {
                    inference_forward_with_tile(
                        &ctx,
                        InferenceFwdOperands {
                            c: typed(&candidate, output_dtype),
                            ..reference_operands
                        },
                        shape,
                        InferenceTile::Sm120Half(tile),
                    )
                    .unwrap_or_else(|error| panic!("exceptional {dtype:?} {tile:?}: {error}"));
                    ctx.stream
                        .synchronize()
                        .expect("exceptional candidate sync");
                    assert_eq!(
                        f32_bits(&ctx, &candidate, shape.m * shape.n),
                        expected,
                        "exceptional {dtype:?}->{output_dtype:?} {tile:?} bias={} changed portable bits",
                        bias_ptr.is_some(),
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet SM120 CUDA device and emits mixed-output TMA data"]
fn half_to_f32_sm120_tma_hot_shapes_smoke() {
    let shapes = [
        InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        // The projection shapes the performance matrix measures: this is
        // where the automatic selector still falls back to the older tensor
        // core tiles instead of the TMA ones.
        InferenceShape {
            m: 2048,
            k: 768,
            n: 3072,
        },
        InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        InferenceShape {
            m: 2048,
            k: 1536,
            n: 768,
        },
        InferenceShape {
            m: 4096,
            k: 3072,
            n: 1536,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("output allocation");
            let cublas = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("cuBLAS allocation");
            let operands = InferenceFwdOperands {
                c: typed(&output, WeightDtype::F32),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let mut auto_tile = InferenceTile::Legacy;
            let auto_us = average_us(&ctx, || {
                auto_tile = inference_forward(
                    &ctx,
                    operands.c,
                    operands.x,
                    operands.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("automatic mixed-output launch");
            });
            let tma_us = InferenceSm120HalfTile::ALL.map(|tile| {
                average_us(&ctx, || {
                    inference_forward_with_tile(
                        &ctx,
                        operands,
                        shape,
                        InferenceTile::Sm120Half(tile),
                    )
                    .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                })
            });
            ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
            let cublas_us = average_us(&ctx, || {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    typed(&cublas, WeightDtype::F32),
                    operands.x,
                    operands.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("fast cuBLAS mixed-output launch");
            });
            ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
            println!(
                "dtype={dtype:?} m={} k={} n={} auto={auto_tile:?} auto_us={auto_us:.3} tma_m64n64_us={:.3} tma_m64n128_us={:.3} tma_m128n64_us={:.3} tma_m128n128_s2_us={:.3} tma_m128n128_s3_us={:.3} cublas_us={cublas_us:.3} auto_over_cublas={:.5}",
                shape.m,
                shape.k,
                shape.n,
                tma_us[0],
                tma_us[1],
                tma_us[2],
                tma_us[3],
                tma_us[4],
                auto_us / cublas_us,
            );
        }
    }
}

#[test]
#[ignore = "requires a quiet 170-SM SM120 CUDA device and screens mixed-output hot B"]
fn half_to_f32_sm120_hot_b_paired_tile_screen() {
    fixed_sm120_tf32_bd_environment_preflight("mixed-output hot B")
        .expect("mixed-output hot B preflight");
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let shape = InferenceShape {
        m: 4621,
        k: 768,
        n: 2304,
    };
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
        let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("candidate allocation");
        let vendor = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("vendor allocation");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa170_0005))
            .expect("A upload");
        b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb170_0005))
            .expect("B upload");
        let candidate_operands = InferenceFwdOperands {
            c: typed(&candidate, WeightDtype::F32),
            x: typed(&a, dtype),
            w: typed(&b, dtype),
            bias_ptr: None,
        };
        let vendor_operands = InferenceFwdOperands {
            c: typed(&vendor, WeightDtype::F32),
            ..candidate_operands
        };
        for tile in InferenceSm120HalfTile::ALL {
            for _ in 0..128 {
                inference_forward_with_tile(
                    &ctx,
                    candidate_operands,
                    shape,
                    InferenceTile::Sm120Half(tile),
                )
                .unwrap_or_else(|error| panic!("warm {tile:?}: {error}"));
            }
            ctx.stream.synchronize().expect("candidate warmup sync");
            configure_fixed_auto_vendor_vendor(&ctx, F32TriadPolicy::ExactScalarFma);
            for _ in 0..128 {
                launch_fixed_auto_vendor_vendor(&ctx, vendor_operands, shape);
            }
            ctx.stream.synchronize().expect("vendor warmup sync");
            let candidate_iterations = fixed_tile_window_iterations(
                &ctx,
                candidate_operands,
                shape,
                InferenceTile::Sm120Half(tile),
            );
            let vendor_iterations =
                fixed_auto_vendor_iterations(fixed_auto_vendor_vendor_window_us(
                    &ctx,
                    vendor_operands,
                    shape,
                    F32TriadPolicy::ExactScalarFma,
                    16,
                ));
            for candidate_first in [true, false] {
                let mut candidate_us = Vec::with_capacity(101);
                let mut vendor_us = Vec::with_capacity(101);
                let mut ratios = Vec::with_capacity(101);
                for _ in 0..101 {
                    let (candidate_elapsed, vendor_elapsed) = if candidate_first {
                        (
                            fixed_tile_window_us(
                                &ctx,
                                candidate_operands,
                                shape,
                                InferenceTile::Sm120Half(tile),
                                candidate_iterations,
                            ),
                            fixed_auto_vendor_vendor_window_us(
                                &ctx,
                                vendor_operands,
                                shape,
                                F32TriadPolicy::ExactScalarFma,
                                vendor_iterations,
                            ),
                        )
                    } else {
                        let vendor_elapsed = fixed_auto_vendor_vendor_window_us(
                            &ctx,
                            vendor_operands,
                            shape,
                            F32TriadPolicy::ExactScalarFma,
                            vendor_iterations,
                        );
                        let candidate_elapsed = fixed_tile_window_us(
                            &ctx,
                            candidate_operands,
                            shape,
                            InferenceTile::Sm120Half(tile),
                            candidate_iterations,
                        );
                        (candidate_elapsed, vendor_elapsed)
                    };
                    candidate_us.push(candidate_elapsed);
                    vendor_us.push(vendor_elapsed);
                    ratios.push(candidate_elapsed / vendor_elapsed);
                }
                candidate_us.sort_by(f64::total_cmp);
                vendor_us.sort_by(f64::total_cmp);
                ratios.sort_by(f64::total_cmp);
                println!(
                    "dtype={dtype:?} tile={tile:?} order={} candidate_p50_us={:.6} vendor_p50_us={:.6} ratio_p50={:.6} ratio_p95={:.6}",
                    if candidate_first {
                        "candidate_then_vendor"
                    } else {
                        "vendor_then_candidate"
                    },
                    percentile(&candidate_us, 0.50),
                    percentile(&vendor_us, 0.50),
                    percentile(&ratios, 0.50),
                    percentile(&ratios, 0.95),
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet 170-SM SM120 CUDA device and diagnoses AUTO launch overhead"]
fn half_to_f32_sm120_hot_b_auto_vs_forced() {
    fixed_sm120_tf32_bd_environment_preflight("mixed-output hot B AUTO versus forced")
        .expect("mixed-output hot B AUTO/forced preflight");
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let shape = InferenceShape {
        m: 4621,
        k: 768,
        n: 2304,
    };
    let tile = InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N128Bk64S2);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
        let auto_output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("AUTO output allocation");
        let forced_output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("forced output allocation");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa170_0005))
            .expect("A upload");
        b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb170_0005))
            .expect("B upload");
        let auto_operands = InferenceFwdOperands {
            c: typed(&auto_output, WeightDtype::F32),
            x: typed(&a, dtype),
            w: typed(&b, dtype),
            bias_ptr: None,
        };
        let forced_operands = InferenceFwdOperands {
            c: typed(&forced_output, WeightDtype::F32),
            ..auto_operands
        };
        for _ in 0..128 {
            launch_fixed_auto_vendor_custom(&ctx, auto_operands, shape);
            inference_forward_with_tile(&ctx, forced_operands, shape, tile)
                .expect("forced warmup launch");
        }
        ctx.stream.synchronize().expect("AUTO/forced warmup sync");
        let auto_iterations = fixed_auto_vendor_iterations(fixed_auto_vendor_custom_window_us(
            &ctx,
            auto_operands,
            shape,
            F32TriadPolicy::ExactScalarFma,
            16,
        ));
        let forced_iterations = fixed_tile_window_iterations(&ctx, forced_operands, shape, tile);
        for auto_first in [true, false] {
            let mut ratios = Vec::with_capacity(101);
            let mut auto_us = Vec::with_capacity(101);
            let mut forced_us = Vec::with_capacity(101);
            for _ in 0..101 {
                let (auto_elapsed, forced_elapsed) = if auto_first {
                    (
                        fixed_auto_vendor_custom_window_us(
                            &ctx,
                            auto_operands,
                            shape,
                            F32TriadPolicy::ExactScalarFma,
                            auto_iterations,
                        ),
                        fixed_tile_window_us(&ctx, forced_operands, shape, tile, forced_iterations),
                    )
                } else {
                    let forced_elapsed =
                        fixed_tile_window_us(&ctx, forced_operands, shape, tile, forced_iterations);
                    let auto_elapsed = fixed_auto_vendor_custom_window_us(
                        &ctx,
                        auto_operands,
                        shape,
                        F32TriadPolicy::ExactScalarFma,
                        auto_iterations,
                    );
                    (auto_elapsed, forced_elapsed)
                };
                auto_us.push(auto_elapsed);
                forced_us.push(forced_elapsed);
                ratios.push(auto_elapsed / forced_elapsed);
            }
            auto_us.sort_by(f64::total_cmp);
            forced_us.sort_by(f64::total_cmp);
            ratios.sort_by(f64::total_cmp);
            println!(
                "dtype={dtype:?} order={} auto_p50_us={:.6} forced_p50_us={:.6} ratio_p50={:.6} ratio_p95={:.6}",
                if auto_first {
                    "auto_then_forced"
                } else {
                    "forced_then_auto"
                },
                percentile(&auto_us, 0.50),
                percentile(&forced_us, 0.50),
                percentile(&ratios, 0.50),
                percentile(&ratios, 0.95),
            );
        }
    }
}

#[test]
#[ignore = "requires a quiet SM80+ CUDA device and emits mixed-output ladder data"]
fn half_to_f32_portable_ladder_smoke() {
    let shapes = [
        InferenceShape {
            m: 1,
            k: 768,
            n: 17,
        },
        InferenceShape { m: 9, k: 33, n: 17 },
        InferenceShape {
            m: 64,
            k: 768,
            n: 24,
        },
        InferenceShape {
            m: 65,
            k: 63,
            n: 127,
        },
        InferenceShape {
            m: 128,
            k: 64,
            n: 128,
        },
        InferenceShape {
            m: 512,
            k: 768,
            n: 512,
        },
        InferenceShape {
            m: 4621,
            k: 384,
            n: 384,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("C allocation");
            let operands = InferenceFwdOperands {
                c: typed(&c, WeightDtype::F32),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            for tile in [
                InferenceTile::Legacy,
                InferenceTile::Tc16,
                InferenceTile::Tc64,
                InferenceTile::Tc128,
            ] {
                let elapsed_us = average_us(&ctx, || {
                    inference_forward_with_tile(&ctx, operands, shape, tile)
                        .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                });
                println!(
                    "dtype={dtype:?} m={} k={} n={} tile={tile:?} elapsed_us={elapsed_us:.3}",
                    shape.m, shape.k, shape.n,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet SM80+ CUDA device and emits mixed-output selector data"]
fn half_to_f32_portable_selector_grid_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for m in [
            1usize, 16, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4621,
        ] {
            for n in [17usize, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2304] {
                let shape = InferenceShape { m, k: 768, n };
                let a =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
                let b =
                    DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
                let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                    .expect("C allocation");
                let operands = InferenceFwdOperands {
                    c: typed(&c, WeightDtype::F32),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr: None,
                };
                for tile in [
                    InferenceTile::Tc16,
                    InferenceTile::Tc64,
                    InferenceTile::Tc128,
                ] {
                    let elapsed_us = average_us(&ctx, || {
                        inference_forward_with_tile(&ctx, operands, shape, tile)
                            .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                    });
                    println!(
                        "dtype={dtype:?} m={m} k=768 n={n} tile={tile:?} elapsed_us={elapsed_us:.3}",
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet SM80+ CUDA device and emits mixed-output K-axis data"]
fn half_to_f32_portable_k_axis_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for m in [2048usize, 4621] {
            for n in [384usize, 768, 1928, 2304] {
                for k in [384usize, 768, 1928, 2304] {
                    let shape = InferenceShape { m, k, n };
                    let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype)
                        .expect("A allocation");
                    let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype)
                        .expect("B allocation");
                    let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                        .expect("C allocation");
                    let operands = InferenceFwdOperands {
                        c: typed(&c, WeightDtype::F32),
                        x: typed(&a, dtype),
                        w: typed(&b, dtype),
                        bias_ptr: None,
                    };
                    for tile in [InferenceTile::Tc64, InferenceTile::Tc128] {
                        let elapsed_us = average_us(&ctx, || {
                            inference_forward_with_tile(&ctx, operands, shape, tile)
                                .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                        });
                        println!(
                            "dtype={dtype:?} m={m} k={k} n={n} tile={tile:?} elapsed_us={elapsed_us:.3}",
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn tcw64_matches_tc128_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in [
            InferenceShape {
                m: 65,
                k: 63,
                n: 127,
            },
            InferenceShape {
                m: 128,
                k: 64,
                n: 128,
            },
            InferenceShape {
                m: 2048,
                k: 2304,
                n: 768,
            },
        ] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let square =
                DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("Tc128 allocation");
            let reuse =
                DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("TcW64 allocation");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa11ce))
                .expect("A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb0b))
                .expect("B upload");
            let operands = InferenceFwdOperands {
                c: typed(&square, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc128)
                .expect("Tc128 launch");
            inference_forward_with_tile(
                &ctx,
                InferenceFwdOperands {
                    c: typed(&reuse, dtype),
                    ..operands
                },
                shape,
                InferenceTile::TcW64,
            )
            .expect("TcW64 launch");
            ctx.stream.synchronize().expect("W64 comparison sync");
            assert_eq!(
                f32_bits(&ctx, &reuse, shape.m * shape.n),
                f32_bits(&ctx, &square, shape.m * shape.n),
                "{dtype:?} TcW64 changed bits for M{} K{} N{}",
                shape.m,
                shape.k,
                shape.n,
            );
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn fixed_portable_half_epilogues_are_tile_and_alignment_invariant() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in [
            InferenceShape {
                m: 15,
                k: 63,
                n: 31,
            },
            InferenceShape {
                m: 17,
                k: 64,
                n: 32,
            },
            InferenceShape {
                m: 65,
                k: 96,
                n: 65,
            },
            InferenceShape {
                m: 129,
                k: 97,
                n: 132,
            },
        ] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let bias =
                DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xe91a))
                .expect("A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xe91b))
                .expect("B upload");
            bias.upload_f32(&ctx.stream, &synth(shape.n, 0xe91c))
                .expect("bias upload");

            for bias_ptr in [None, Some(bias.cached_ptr())] {
                for output_offset in [0usize, 1] {
                    let elements = shape.m * shape.n;
                    let reference = DtypedBuf::zeros(&ctx.stream, elements + 1, dtype)
                        .expect("reference allocation");
                    let reference_ptr = TypedPtr {
                        ptr: reference.cached_ptr() + (output_offset * dtype.size_bytes()) as u64,
                        dtype,
                    };
                    let base_operands = InferenceFwdOperands {
                        c: reference_ptr,
                        x: typed(&a, dtype),
                        w: typed(&b, dtype),
                        bias_ptr,
                    };
                    inference_forward_with_tile(&ctx, base_operands, shape, InferenceTile::Tc128)
                        .expect("Tc128 reference");
                    ctx.stream.synchronize().expect("reference sync");
                    let expected = f32_bits(&ctx, &reference, elements + 1)[output_offset..]
                        [..elements]
                        .to_vec();

                    for tile in [InferenceTile::Tc16, InferenceTile::Tc64] {
                        let output = DtypedBuf::zeros(&ctx.stream, elements + 1, dtype)
                            .expect("candidate allocation");
                        let operands = InferenceFwdOperands {
                            c: TypedPtr {
                                ptr: output.cached_ptr()
                                    + (output_offset * dtype.size_bytes()) as u64,
                                dtype,
                            },
                            ..base_operands
                        };
                        inference_forward_with_tile(&ctx, operands, shape, tile)
                            .unwrap_or_else(|error| panic!("{tile:?} launch: {error}"));
                        ctx.stream.synchronize().expect("candidate sync");
                        assert_eq!(
                            f32_bits(&ctx, &output, elements + 1)[output_offset..][..elements],
                            expected,
                            "{dtype:?} {tile:?} epilogue drift for {shape:?} bias={} offset={output_offset}",
                            bias_ptr.is_some()
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_half_tma_matches_portable_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in [
            InferenceShape {
                m: 65,
                k: 40,
                n: 72,
            },
            InferenceShape {
                m: 129,
                k: 96,
                n: 136,
            },
        ] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let bias =
                DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x120a))
                .expect("A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x120b))
                .expect("B upload");
            bias.upload_f32(&ctx.stream, &synth(shape.n, 0x120c))
                .expect("bias upload");
            for bias_ptr in [None, Some(bias.cached_ptr())] {
                let reference =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("reference");
                let reference_operands = InferenceFwdOperands {
                    c: typed(&reference, dtype),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr,
                };
                inference_forward_with_tile(&ctx, reference_operands, shape, InferenceTile::Tc128)
                    .expect("portable reference launch");
                ctx.stream.synchronize().expect("portable reference sync");
                let expected = f32_bits(&ctx, &reference, shape.m * shape.n);
                for candidate in InferenceSm120HalfTile::ALL {
                    let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype)
                        .expect("candidate output");
                    let operands = InferenceFwdOperands {
                        c: typed(&output, dtype),
                        ..reference_operands
                    };
                    let tile = InferenceTile::Sm120Half(candidate);
                    inference_forward_with_tile(&ctx, operands, shape, tile)
                        .unwrap_or_else(|error| panic!("first {candidate:?}: {error}"));
                    ctx.stream.synchronize().expect("first candidate sync");
                    let first = f32_bits(&ctx, &output, shape.m * shape.n);
                    inference_forward_with_tile(&ctx, operands, shape, tile)
                        .unwrap_or_else(|error| panic!("repeat {candidate:?}: {error}"));
                    ctx.stream.synchronize().expect("repeat candidate sync");
                    let second = f32_bits(&ctx, &output, shape.m * shape.n);
                    assert_eq!(first, second, "repeat drift for {dtype:?} {candidate:?}");
                    assert_eq!(
                        first,
                        expected,
                        "portable drift for {dtype:?} {shape:?} {candidate:?} bias={}",
                        bias_ptr.is_some(),
                    );
                }
            }
        }
    }
}

fn fixed_sm120_half_symbol(tile: InferenceSm120HalfTile, dtype: WeightDtype) -> &'static str {
    match (tile, dtype) {
        (InferenceSm120HalfTile::M64N64Bk64S2, WeightDtype::Bf16) => {
            "nn_sm120_tma_64x64_bk64_s2_bf16"
        }
        (InferenceSm120HalfTile::M64N64Bk64S2, WeightDtype::F16) => {
            "nn_sm120_tma_64x64_bk64_s2_f16"
        }
        (InferenceSm120HalfTile::M64N128Bk64S2, WeightDtype::Bf16) => {
            "nn_sm120_tma_64x128_bk64_s2_bf16"
        }
        (InferenceSm120HalfTile::M64N128Bk64S2, WeightDtype::F16) => {
            "nn_sm120_tma_64x128_bk64_s2_f16"
        }
        (InferenceSm120HalfTile::M128N64Bk32S3, WeightDtype::Bf16) => {
            "nn_sm120_tma_128x64_bk32_s3_bf16"
        }
        (InferenceSm120HalfTile::M128N64Bk32S3, WeightDtype::F16) => {
            "nn_sm120_tma_128x64_bk32_s3_f16"
        }
        (InferenceSm120HalfTile::M128N128Bk32S2, WeightDtype::Bf16) => {
            "nn_sm120_tma_128x128_bk32_s2_bf16"
        }
        (InferenceSm120HalfTile::M128N128Bk32S2, WeightDtype::F16) => {
            "nn_sm120_tma_128x128_bk32_s2_f16"
        }
        (InferenceSm120HalfTile::M128N128Bk32S3, WeightDtype::Bf16) => {
            "nn_sm120_tma_128x128_bk32_s3_bf16"
        }
        (InferenceSm120HalfTile::M128N128Bk32S3, WeightDtype::F16) => {
            "nn_sm120_tma_128x128_bk32_s3_f16"
        }
        (_, WeightDtype::F32) => panic!("SM120 half symbol requested for F32"),
    }
}

fn run_fixed_sm120_half_exact_overlay_route_case(
    ctx: &GpuCtx,
    label: &str,
    dtype: WeightDtype,
    shape: InferenceShape,
    has_bias: bool,
    promoted: InferenceSm120HalfTile,
    incumbent: InferenceSm120HalfTile,
) {
    let elements = shape.m * shape.n;
    let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
    let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
    let bias = DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
    a.upload_f32(
        &ctx.stream,
        &synth(shape.m * shape.k, 0xa132 ^ shape.m as u64),
    )
    .expect("A upload");
    b.upload_f32(
        &ctx.stream,
        &synth(shape.k * shape.n, 0xb132 ^ shape.n as u64),
    )
    .expect("B upload");
    bias.upload_f32(&ctx.stream, &synth(shape.n, 0xb1a5 ^ shape.k as u64))
        .expect("bias upload");
    let bias_ptr = has_bias.then(|| bias.cached_ptr());

    let forced = DtypedBuf::zeros(&ctx.stream, elements, dtype).expect("forced output");
    let production = DtypedBuf::zeros(&ctx.stream, elements, dtype).expect("AUTO output");
    let forced_operands = InferenceFwdOperands {
        c: typed(&forced, dtype),
        x: typed(&a, dtype),
        w: typed(&b, dtype),
        bias_ptr,
    };
    let production_operands = InferenceFwdOperands {
        c: typed(&production, dtype),
        ..forced_operands
    };
    inference_forward_with_tile(
        ctx,
        forced_operands,
        shape,
        InferenceTile::Sm120Half(incumbent),
    )
    .expect("forced pre-overlay incumbent");
    let selected = inference_forward(
        ctx,
        production_operands.c,
        production_operands.x,
        production_operands.w,
        bias_ptr,
        (shape.m, shape.k, shape.n),
    )
    .expect("production AUTO launch");
    assert_eq!(
        selected,
        InferenceTile::Sm120Half(promoted),
        "{label} AUTO tile"
    );
    ctx.stream.synchronize().expect("eager synchronization");
    let expected = f32_bits(ctx, &forced, elements);
    assert_eq!(
        f32_bits(ctx, &production, elements),
        expected,
        "{label} eager bits"
    );

    let repeated = inference_forward(
        ctx,
        production_operands.c,
        production_operands.x,
        production_operands.w,
        bias_ptr,
        (shape.m, shape.k, shape.n),
    )
    .expect("production AUTO repeat");
    assert_eq!(
        repeated,
        InferenceTile::Sm120Half(promoted),
        "{label} repeat tile"
    );
    ctx.stream.synchronize().expect("repeat synchronization");
    assert_eq!(
        f32_bits(ctx, &production, elements),
        expected,
        "{label} repeat bits"
    );

    let graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            inference_forward(
                ctx,
                production_operands.c,
                production_operands.x,
                production_operands.w,
                bias_ptr,
                (shape.m, shape.k, shape.n),
            )
            .and_then(|tile| {
                (tile == InferenceTile::Sm120Half(promoted))
                    .then_some(())
                    .ok_or_else(|| format!("{label} graph selected {tile:?}"))
            })
        })
    }
    .expect("capture production AUTO graph");
    assert_eq!(
        single_graph_kernel_name(&graph, label),
        fixed_sm120_half_symbol(promoted, dtype),
        "{label} promoted physical route"
    );
    for replay in 0..10 {
        graph.launch().expect("production graph replay");
        ctx.stream.synchronize().expect("graph synchronization");
        assert_eq!(
            f32_bits(ctx, &production, elements),
            expected,
            "{label} graph replay {replay} bits"
        );
    }

    let misaligned =
        DtypedBuf::zeros(&ctx.stream, elements + 1, dtype).expect("misaligned AUTO output");
    let misaligned_operands = InferenceFwdOperands {
        c: TypedPtr {
            ptr: misaligned.cached_ptr() + dtype.size_bytes() as u64,
            dtype,
        },
        ..forced_operands
    };
    assert_eq!(misaligned_operands.c.ptr & 3, 2, "{label} C+2 setup");
    let misaligned_tile = inference_forward(
        ctx,
        misaligned_operands.c,
        misaligned_operands.x,
        misaligned_operands.w,
        bias_ptr,
        (shape.m, shape.k, shape.n),
    )
    .expect("misaligned AUTO launch");
    assert_eq!(
        misaligned_tile,
        InferenceTile::Sm120Half(incumbent),
        "{label} C+2 fallback tile"
    );
    ctx.stream
        .synchronize()
        .expect("misaligned eager synchronization");
    let misaligned_bits = f32_bits(ctx, &misaligned, elements + 1);
    assert_eq!(&misaligned_bits[1..], expected, "{label} C+2 fallback bits");
    let misaligned_graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            inference_forward(
                ctx,
                misaligned_operands.c,
                misaligned_operands.x,
                misaligned_operands.w,
                bias_ptr,
                (shape.m, shape.k, shape.n),
            )
            .map(|_| ())
        })
    }
    .expect("capture C+2 fallback graph");
    assert_eq!(
        single_graph_kernel_name(&misaligned_graph, label),
        fixed_sm120_half_symbol(incumbent, dtype),
        "{label} C+2 physical fallback"
    );
}

#[test]
#[ignore = "requires an RTX5090 with the Fixed module loaded by NVRTC 13.2"]
fn fixed_sm120_half_exact_selector_routes_match_incumbent_bits_and_graphs() {
    use InferenceSm120HalfTile::{
        M64N64Bk64S2 as C, M64N128Bk64S2 as D, M128N64Bk32S3 as A, M128N128Bk32S2 as B,
    };

    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert_eq!(ctx.kernels.compiler_identity().nvrtc_version, (13, 2));
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);

    for (label, dtype, shape, has_bias, expected) in [
        (
            "A1_bf16_none_rejected",
            WeightDtype::Bf16,
            InferenceShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            false,
            C,
        ),
        (
            "A1_bf16_bias_rejected",
            WeightDtype::Bf16,
            InferenceShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            true,
            C,
        ),
        (
            "A1_f16_none_rejected",
            WeightDtype::F16,
            InferenceShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            false,
            C,
        ),
        (
            "A1_f16_bias_rejected",
            WeightDtype::F16,
            InferenceShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            true,
            C,
        ),
        (
            "A2_bf16_bias_rejected",
            WeightDtype::Bf16,
            InferenceShape {
                m: 4096,
                k: 384,
                n: 1536,
            },
            true,
            C,
        ),
        (
            "A2_f16_bias_rejected",
            WeightDtype::F16,
            InferenceShape {
                m: 4096,
                k: 384,
                n: 1536,
            },
            true,
            C,
        ),
        (
            "A3_f16_none_rejected",
            WeightDtype::F16,
            InferenceShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            false,
            C,
        ),
        (
            "A3_f16_bias_rejected",
            WeightDtype::F16,
            InferenceShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            true,
            C,
        ),
        (
            "B_bf16_none_promoted",
            WeightDtype::Bf16,
            InferenceShape {
                m: 3072,
                k: 1928,
                n: 1928,
            },
            false,
            D,
        ),
        (
            "B_bf16_bias_promoted",
            WeightDtype::Bf16,
            InferenceShape {
                m: 3072,
                k: 1928,
                n: 1928,
            },
            true,
            D,
        ),
        (
            "B_f16_none_promoted",
            WeightDtype::F16,
            InferenceShape {
                m: 3072,
                k: 1928,
                n: 1928,
            },
            false,
            D,
        ),
        (
            "B_f16_bias_promoted",
            WeightDtype::F16,
            InferenceShape {
                m: 3072,
                k: 1928,
                n: 1928,
            },
            true,
            D,
        ),
        (
            "A2_bf16_none_rejected",
            WeightDtype::Bf16,
            InferenceShape {
                m: 4096,
                k: 384,
                n: 1536,
            },
            false,
            C,
        ),
        (
            "A2_f16_none_rejected",
            WeightDtype::F16,
            InferenceShape {
                m: 4096,
                k: 384,
                n: 1536,
            },
            false,
            C,
        ),
        (
            "A3_bf16_none_rejected",
            WeightDtype::Bf16,
            InferenceShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            false,
            C,
        ),
        (
            "A3_bf16_bias_rejected",
            WeightDtype::Bf16,
            InferenceShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            true,
            C,
        ),
    ] {
        run_fixed_sm120_half_exact_overlay_route_case(
            &ctx, label, dtype, shape, has_bias, expected, C,
        );
    }
    for (shape_name, shape, incumbent) in [
        (
            "m1024_k1928_n1928",
            InferenceShape {
                m: 1024,
                k: 1928,
                n: 1928,
            },
            B,
        ),
        (
            "m1024_k1928_n2304",
            InferenceShape {
                m: 1024,
                k: 1928,
                n: 2304,
            },
            B,
        ),
        (
            "m1536_k1032_n1536",
            InferenceShape {
                m: 1536,
                k: 1032,
                n: 1536,
            },
            B,
        ),
        (
            "m1536_k1928_n1536",
            InferenceShape {
                m: 1536,
                k: 1928,
                n: 1536,
            },
            B,
        ),
        (
            "m4621_k1928_n1928",
            InferenceShape {
                m: 4621,
                k: 1928,
                n: 1928,
            },
            B,
        ),
        (
            "m1536_k768_n1536",
            InferenceShape {
                m: 1536,
                k: 768,
                n: 1536,
            },
            A,
        ),
    ] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for has_bias in [false, true] {
                let label = format!(
                    "{shape_name}_{}_{}",
                    dtype.as_str(),
                    if has_bias { "bias" } else { "none" }
                );
                // The qualified no-bias S3 cells supersede D only for these
                // exact dtype/shape combinations; all bit/graph gates remain.
                let expected = if !has_bias
                    && ((shape.m, shape.k, shape.n) == (1536, 1032, 1536)
                        || (dtype == WeightDtype::F16
                            && matches!(
                                (shape.m, shape.k, shape.n),
                                (1024, 1928, 1928) | (1024, 1928, 2304) | (1536, 1928, 1536)
                            ))) {
                    InferenceSm120HalfTile::M128N128Bk32S3
                } else {
                    D
                };
                run_fixed_sm120_half_exact_overlay_route_case(
                    &ctx, &label, dtype, shape, has_bias, expected, incumbent,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_half_retained_tiles_guarded_tails_are_bit_exact() {
    const GUARD: usize = 64;
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert!(
        matches!(device.compute_capability, (12, 0) | (12, 1)),
        "SM120 or SM121 required",
    );
    assert!(
        ctx.kernels.gemm_bi_nn_half_sm120.is_some(),
        "Fixed SM120 half module must be loaded",
    );
    assert_eq!(
        InferenceSm120HalfTile::ALL,
        [
            InferenceSm120HalfTile::M64N64Bk64S2,
            InferenceSm120HalfTile::M64N128Bk64S2,
            InferenceSm120HalfTile::M128N64Bk32S3,
            InferenceSm120HalfTile::M128N128Bk32S2,
            InferenceSm120HalfTile::M128N128Bk32S3,
        ],
    );
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);

    for (tile, shape, geometry) in [
        (
            InferenceSm120HalfTile::M64N64Bk64S2,
            InferenceShape {
                m: 65,
                k: 136,
                n: 72,
            },
            (64usize, 64usize, 64usize, 2usize),
        ),
        (
            InferenceSm120HalfTile::M128N64Bk32S3,
            InferenceShape {
                m: 129,
                k: 104,
                n: 72,
            },
            (128, 64, 32, 3),
        ),
        (
            InferenceSm120HalfTile::M64N128Bk64S2,
            InferenceShape {
                m: 65,
                k: 136,
                n: 136,
            },
            (64, 128, 64, 2),
        ),
        (
            InferenceSm120HalfTile::M128N128Bk32S2,
            InferenceShape {
                m: 129,
                k: 72,
                n: 136,
            },
            (128, 128, 32, 2),
        ),
        (
            InferenceSm120HalfTile::M128N128Bk32S3,
            InferenceShape {
                m: 129,
                k: 104,
                n: 136,
            },
            (128, 128, 32, 3),
        ),
    ] {
        let (bm, bn, bk, stages) = geometry;
        assert_ne!(shape.m % bm, 0);
        assert_ne!(shape.k % bk, 0);
        assert_ne!(shape.n % bn, 0);
        assert!(shape.k.div_ceil(bk) > stages);

        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let a_len = shape.m * shape.k;
            let b_len = shape.k * shape.n;
            let output_len = shape.m * shape.n;
            let mut a_host = vec![13.0f32; GUARD + a_len + GUARD];
            let mut b_host = vec![-11.0f32; GUARD + b_len + GUARD];
            a_host[GUARD..GUARD + a_len].copy_from_slice(&synth(a_len, 0xa11ce ^ shape.m as u64));
            b_host[GUARD..GUARD + b_len].copy_from_slice(&synth(b_len, 0xb0b ^ shape.n as u64));
            let a = DtypedBuf::zeros(&ctx.stream, a_host.len(), dtype).expect("guarded A");
            let b = DtypedBuf::zeros(&ctx.stream, b_host.len(), dtype).expect("guarded B");
            a.upload_f32(&ctx.stream, &a_host).expect("A upload");
            b.upload_f32(&ctx.stream, &b_host).expect("B upload");
            let a_ptr = a.cached_ptr() + (GUARD * dtype.size_bytes()) as u64;
            let b_ptr = b.cached_ptr() + (GUARD * dtype.size_bytes()) as u64;
            assert_eq!(a_ptr & 15, 0);
            assert_eq!(b_ptr & 15, 0);
            let mut a_before = vec![0.0f32; a_host.len()];
            let mut b_before = vec![0.0f32; b_host.len()];
            a.download_f32(&ctx.stream, &mut a_before)
                .expect("A snapshot");
            b.download_f32(&ctx.stream, &mut b_before)
                .expect("B snapshot");

            let mut bias_host = vec![23.5f32; GUARD + shape.n + GUARD];
            bias_host[GUARD..GUARD + shape.n]
                .copy_from_slice(&synth(shape.n, 0xb1a5 ^ shape.k as u64));
            let bias = DtypedBuf::zeros(&ctx.stream, bias_host.len(), WeightDtype::F32)
                .expect("guarded bias");
            bias.upload_f32(&ctx.stream, &bias_host)
                .expect("bias upload");
            let bias_ptr = bias.cached_ptr() + (GUARD * WeightDtype::F32.size_bytes()) as u64;
            let mut bias_before = vec![0.0f32; bias_host.len()];
            bias.download_f32(&ctx.stream, &mut bias_before)
                .expect("bias snapshot");
            let mut unbiased_reference: Option<Vec<u32>> = None;

            for candidate_bias in [None, Some(bias_ptr)] {
                let reference =
                    DtypedBuf::zeros(&ctx.stream, output_len, dtype).expect("portable reference");
                let reference_operands = InferenceFwdOperands {
                    c: typed(&reference, dtype),
                    x: TypedPtr { ptr: a_ptr, dtype },
                    w: TypedPtr { ptr: b_ptr, dtype },
                    bias_ptr: candidate_bias,
                };
                inference_forward_with_tile(&ctx, reference_operands, shape, InferenceTile::Tc128)
                    .expect("portable reference launch");
                ctx.stream.synchronize().expect("portable reference sync");
                let expected = f32_bits(&ctx, &reference, output_len);
                assert!(expected.iter().any(|&bits| bits != 0));
                if let Some(unbiased) = &unbiased_reference {
                    assert!(
                        expected
                            .iter()
                            .zip(unbiased)
                            .any(|(left, right)| left != right),
                        "bias must change at least one output for {dtype:?} {tile:?}",
                    );
                } else {
                    unbiased_reference = Some(expected.clone());
                }

                for offset in [0usize, GUARD + 1] {
                    let storage_len = offset + output_len + GUARD;
                    let candidate = DtypedBuf::zeros(&ctx.stream, storage_len, dtype)
                        .expect("guarded candidate output");
                    let candidate_ptr =
                        candidate.cached_ptr() + (offset * dtype.size_bytes()) as u64;
                    if offset == 0 {
                        assert_eq!(candidate_ptr & 3, 0);
                    } else {
                        assert_eq!(candidate_ptr & 3, 2);
                    }
                    let candidate_operands = InferenceFwdOperands {
                        c: TypedPtr {
                            ptr: candidate_ptr,
                            dtype,
                        },
                        ..reference_operands
                    };
                    let mut first_bits = None;
                    for poison in [-7.0f32, 9.0f32] {
                        let mut reset = vec![19.25f32; storage_len];
                        reset[offset..offset + output_len].fill(poison);
                        candidate
                            .upload_f32(&ctx.stream, &reset)
                            .expect("candidate reset");
                        inference_forward_with_tile(
                            &ctx,
                            candidate_operands,
                            shape,
                            InferenceTile::Sm120Half(tile),
                        )
                        .expect("SM120 guarded launch");
                        ctx.stream.synchronize().expect("SM120 guarded sync");
                        let mut observed = vec![0.0f32; storage_len];
                        candidate
                            .download_f32(&ctx.stream, &mut observed)
                            .expect("candidate download");
                        assert!(
                            observed[..offset]
                                .iter()
                                .all(|value| value.to_bits() == 19.25f32.to_bits())
                        );
                        assert!(
                            observed[offset + output_len..]
                                .iter()
                                .all(|value| value.to_bits() == 19.25f32.to_bits())
                        );
                        let logical_bits = observed[offset..offset + output_len]
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>();
                        assert_eq!(logical_bits, expected, "{dtype:?} {tile:?} offset={offset}");
                        if let Some(first) = &first_bits {
                            assert_eq!(&logical_bits, first, "repeat drift for {dtype:?} {tile:?}");
                        } else {
                            first_bits = Some(logical_bits);
                        }
                        let mut a_after = vec![0.0f32; a_before.len()];
                        let mut b_after = vec![0.0f32; b_before.len()];
                        let mut bias_after = vec![0.0f32; bias_before.len()];
                        a.download_f32(&ctx.stream, &mut a_after).expect("A verify");
                        b.download_f32(&ctx.stream, &mut b_after).expect("B verify");
                        bias.download_f32(&ctx.stream, &mut bias_after)
                            .expect("bias verify");
                        assert_eq!(a_after, a_before, "A mutated for {dtype:?} {tile:?}");
                        assert_eq!(b_after, b_before, "B mutated for {dtype:?} {tile:?}");
                        assert_eq!(
                            bias_after, bias_before,
                            "bias mutated for {dtype:?} {tile:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_half_graph_replay_is_bit_exact_and_cold_capture_fails() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    let dtype = WeightDtype::Bf16;
    let shape = InferenceShape {
        m: 128,
        k: 96,
        n: 1536,
    };
    let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
    let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
    let output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("output");
    a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xc120a))
        .expect("A upload");
    b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xc120b))
        .expect("B upload");
    let run = || {
        inference_forward(
            &ctx,
            typed(&output, dtype),
            typed(&a, dtype),
            typed(&b, dtype),
            None,
            (shape.m, shape.k, shape.n),
        )
        .and_then(|tile| match tile {
            InferenceTile::Sm120Half(_) => Ok(()),
            _ => Err(format!("unexpected production tile {tile:?}")),
        })
    };
    run().expect("warm half tensor maps");
    ctx.stream.synchronize().expect("eager synchronization");
    let eager = f32_bits(&ctx, &output, shape.m * shape.n);
    let graph = unsafe { capture_into_graph(&ctx.stream, run) }.expect("capture Fixed SM120 half");
    for replay in 0..10 {
        graph.launch().expect("launch Fixed SM120 half graph");
        ctx.stream.synchronize().expect("graph synchronization");
        assert_eq!(
            f32_bits(&ctx, &output, shape.m * shape.n),
            eager,
            "Fixed SM120 half graph replay {replay} changed bits"
        );
    }

    let cold_a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("cold A");
    let cold_b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("cold B");
    let cold_capture = unsafe {
        capture_into_graph(&ctx.stream, || {
            inference_forward(
                &ctx,
                typed(&output, dtype),
                typed(&cold_a, dtype),
                typed(&cold_b, dtype),
                None,
                (shape.m, shape.k, shape.n),
            )
            .map(|_| ())
        })
    };
    let error = match cold_capture {
        Ok(_) => panic!("cold half tensor maps must not be encoded during capture"),
        Err(error) => error,
    };
    assert!(
        error.contains("must be prepared before graph capture"),
        "unexpected cold-capture error: {error}"
    );
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_half_graph_warmup_retains_more_than_32_live_maps() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    let shape = InferenceShape {
        m: 128,
        k: 96,
        n: 1536,
    };

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
        let output =
            DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("output allocation");
        a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xcac4e))
            .expect("A upload");
        let mut weights = Vec::with_capacity(33);
        for index in 0..33_u64 {
            let weight =
                DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("weight allocation");
            weight
                .upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x120_000 + index))
                .expect("weight upload");
            weights.push(weight);
        }

        for weight in &weights {
            let tile = inference_forward(
                &ctx,
                typed(&output, dtype),
                typed(&a, dtype),
                typed(weight, dtype),
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("warm tensor map");
            assert!(matches!(tile, InferenceTile::Sm120Half(_)));
        }
        ctx.stream.synchronize().expect("warmup synchronization");
        let eager = f32_bits(&ctx, &output, shape.m * shape.n);

        let graph = unsafe {
            capture_into_graph(&ctx.stream, || {
                for weight in &weights {
                    let tile = inference_forward(
                        &ctx,
                        typed(&output, dtype),
                        typed(&a, dtype),
                        typed(weight, dtype),
                        None,
                        (shape.m, shape.k, shape.n),
                    )?;
                    if !matches!(tile, InferenceTile::Sm120Half(_)) {
                        return Err(format!("unexpected production tile {tile:?}"));
                    }
                }
                Ok(())
            })
        }
        .expect("capture more than 32 live tensor maps");
        graph.launch().expect("launch multi-map graph");
        ctx.stream.synchronize().expect("multi-map graph sync");
        assert_eq!(
            f32_bits(&ctx, &output, shape.m * shape.n),
            eager,
            "{dtype:?} graph replay changed bits after 33 live map keys"
        );
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_sm120_half_is_batch_prefix_invariant_across_selectors() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    let (small_m, large_m, k, n) = (64usize, 512usize, 384usize, 1536usize);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let large_a_host = synth(large_m * k, 0xba7c4);
        let b_host = synth(k * n, 0xb32);
        let bias_host = synth(n, 0xb1a5);
        let small_a = DtypedBuf::zeros(&ctx.stream, small_m * k, dtype).expect("small A");
        let large_a = DtypedBuf::zeros(&ctx.stream, large_m * k, dtype).expect("large A");
        let b = DtypedBuf::zeros(&ctx.stream, k * n, dtype).expect("B");
        let bias = DtypedBuf::zeros(&ctx.stream, n, WeightDtype::F32).expect("bias");
        let small_c = DtypedBuf::zeros(&ctx.stream, small_m * n, dtype).expect("small C");
        let large_c = DtypedBuf::zeros(&ctx.stream, large_m * n, dtype).expect("large C");
        small_a
            .upload_f32(&ctx.stream, &large_a_host[..small_m * k])
            .expect("small A upload");
        large_a
            .upload_f32(&ctx.stream, &large_a_host)
            .expect("large A upload");
        b.upload_f32(&ctx.stream, &b_host).expect("B upload");
        bias.upload_f32(&ctx.stream, &bias_host)
            .expect("bias upload");
        let small_tile = inference_forward(
            &ctx,
            typed(&small_c, dtype),
            typed(&small_a, dtype),
            typed(&b, dtype),
            Some(bias.cached_ptr()),
            (small_m, k, n),
        )
        .expect("small launch");
        let large_tile = inference_forward(
            &ctx,
            typed(&large_c, dtype),
            typed(&large_a, dtype),
            typed(&b, dtype),
            Some(bias.cached_ptr()),
            (large_m, k, n),
        )
        .expect("large launch");
        ctx.stream.synchronize().expect("prefix sync");
        assert_ne!(small_tile, large_tile, "test must cross selector rungs");
        assert!(
            matches!(large_tile, InferenceTile::Sm120Half(_)),
            "large shape did not select SM120 TMA: {large_tile:?}"
        );
        assert_eq!(
            f32_bits(&ctx, &small_c, small_m * n),
            f32_bits(&ctx, &large_c, large_m * n)[..small_m * n],
            "{dtype:?} prefix bits changed across selectors"
        );
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_half_narrow_n_uses_portable_numeric_family() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    let (small_m, large_m, k, n) = (16usize, 8192usize, 768usize, 24usize);

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let large_a_host = synth(large_m * k, 0xba7c4);
        let b_host = synth(k * n, 0xb32);
        let bias_host = synth(n, 0xb1a5);
        let small_a = DtypedBuf::zeros(&ctx.stream, small_m * k, dtype).expect("small A");
        let large_a = DtypedBuf::zeros(&ctx.stream, large_m * k, dtype).expect("large A");
        let b = DtypedBuf::zeros(&ctx.stream, k * n, dtype).expect("B");
        let bias = DtypedBuf::zeros(&ctx.stream, n, WeightDtype::F32).expect("bias");
        small_a
            .upload_f32(&ctx.stream, &large_a_host[..small_m * k])
            .expect("small A upload");
        large_a
            .upload_f32(&ctx.stream, &large_a_host)
            .expect("large A upload");
        b.upload_f32(&ctx.stream, &b_host).expect("B upload");
        bias.upload_f32(&ctx.stream, &bias_host)
            .expect("bias upload");

        for bias_ptr in [None, Some(bias.cached_ptr())] {
            let small_c = DtypedBuf::zeros(&ctx.stream, small_m * n, dtype).expect("small C");
            let large_c = DtypedBuf::zeros(&ctx.stream, large_m * n, dtype).expect("large C");
            let small_tile = inference_forward(
                &ctx,
                typed(&small_c, dtype),
                typed(&small_a, dtype),
                typed(&b, dtype),
                bias_ptr,
                (small_m, k, n),
            )
            .expect("small narrow-N launch");
            let large_tile = inference_forward(
                &ctx,
                typed(&large_c, dtype),
                typed(&large_a, dtype),
                typed(&b, dtype),
                bias_ptr,
                (large_m, k, n),
            )
            .expect("large narrow-N launch");
            ctx.stream.synchronize().expect("narrow-N prefix sync");
            assert_eq!(small_tile, InferenceTile::Tc16);
            assert_eq!(large_tile, InferenceTile::Tc16);
            assert_eq!(
                f32_bits(&ctx, &small_c, small_m * n),
                f32_bits(&ctx, &large_c, large_m * n)[..small_m * n],
                "{dtype:?} narrow-N prefix bits changed with bias={}",
                bias_ptr.is_some()
            );
        }
    }
}

#[test]
#[ignore = "requires a quiet SM80+ CUDA device and emits narrow-N performance data"]
fn fixed_half_narrow_n_performance_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in [
            InferenceShape {
                m: 1,
                k: 768,
                n: 17,
            },
            InferenceShape {
                m: 16,
                k: 768,
                n: 24,
            },
            InferenceShape {
                m: 64,
                k: 768,
                n: 24,
            },
            InferenceShape {
                m: 512,
                k: 768,
                n: 24,
            },
            InferenceShape {
                m: 8192,
                k: 768,
                n: 24,
            },
        ] {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let legacy =
                DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("legacy allocation");
            let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype)
                .expect("candidate allocation");
            let vendor =
                DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("vendor allocation");
            let legacy_ops = InferenceFwdOperands {
                c: typed(&legacy, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let candidate_ops = InferenceFwdOperands {
                c: typed(&candidate, dtype),
                ..legacy_ops
            };
            let vendor_ops = InferenceFwdOperands {
                c: typed(&vendor, dtype),
                ..legacy_ops
            };
            let legacy_us = average_us(&ctx, || {
                inference_forward_with_tile(&ctx, legacy_ops, shape, InferenceTile::Legacy)
                    .expect("legacy launch");
            });
            let tc16_us = average_us(&ctx, || {
                inference_forward_with_tile(&ctx, candidate_ops, shape, InferenceTile::Tc16)
                    .expect("Tc16 launch");
            });
            ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
            let cublas_us = average_us(&ctx, || {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    vendor_ops.c,
                    vendor_ops.x,
                    vendor_ops.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("fast cuBLAS launch");
            });
            ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
            println!(
                "dtype={dtype:?} m={} k={} n={} legacy_us={legacy_us:.3} tc16_us={tc16_us:.3} cublas_us={cublas_us:.3} tc16_over_legacy={:.5} tc16_over_cublas={:.5}",
                shape.m,
                shape.k,
                shape.n,
                tc16_us / legacy_us,
                tc16_us / cublas_us,
            );
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device and emits performance data"]
fn fixed_sm120_half_hot_census() {
    let shapes = [
        InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
            let operands = InferenceFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let portable_us = average_us(&ctx, || {
                inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc128)
                    .expect("portable launch");
            });
            ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
            let cublas_us = average_us(&ctx, || {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    operands.c,
                    operands.x,
                    operands.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("cuBLAS launch");
            });
            ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
            for candidate in InferenceSm120HalfTile::ALL {
                let elapsed_us = average_us(&ctx, || {
                    inference_forward_with_tile(
                        &ctx,
                        operands,
                        shape,
                        InferenceTile::Sm120Half(candidate),
                    )
                    .unwrap_or_else(|error| panic!("forced {candidate:?}: {error}"));
                });
                println!(
                    "dtype={dtype:?} m={} k={} n={} tile={candidate:?} elapsed_us={elapsed_us:.3} portable_us={portable_us:.3} cublas_us={cublas_us:.3} over_portable={:.5} over_cublas={:.5}",
                    shape.m,
                    shape.k,
                    shape.n,
                    elapsed_us / portable_us,
                    elapsed_us / cublas_us,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device and emits selector data"]
fn fixed_sm120_half_selector_census() {
    let candidates = [
        InferenceSm120HalfTile::M64N64Bk64S2,
        InferenceSm120HalfTile::M128N64Bk32S3,
        InferenceSm120HalfTile::M128N128Bk32S2,
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    let dtype = WeightDtype::Bf16;
    for m in [16usize, 64, 128] {
        for k in [64usize, 384, 768, 1928] {
            for n in [384usize, 1536] {
                run_fixed_sm120_half_selector_cell(
                    &ctx,
                    dtype,
                    InferenceShape { m, k, n },
                    candidates,
                );
            }
        }
    }
    for m in [256usize, 512, 1024] {
        for k in [384usize, 768, 1928] {
            for n in [384usize, 1536, 2304] {
                run_fixed_sm120_half_selector_cell(
                    &ctx,
                    dtype,
                    InferenceShape { m, k, n },
                    candidates,
                );
            }
        }
    }
    for m in [1536usize, 2048, 3072, 4096, 4621] {
        for k in [384usize, 768, 1928] {
            for n in [1536usize, 1928, 2304] {
                run_fixed_sm120_half_selector_cell(
                    &ctx,
                    dtype,
                    InferenceShape { m, k, n },
                    candidates,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet 170-SM CC12 CUDA device and emits paired selector evidence"]
fn fixed_sm120_half_selector_paired_gaps() {
    use InferenceSm120HalfTile::M128N128Bk32S2 as B;
    use InferenceSm120HalfTile::{M64N64Bk64S2 as C, M128N64Bk32S3 as A};

    let cells = [
        (
            InferenceShape {
                m: 512,
                k: 1928,
                n: 2304,
            },
            B,
            C,
        ),
        (
            InferenceShape {
                m: 1536,
                k: 384,
                n: 1536,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 1536,
                k: 768,
                n: 1536,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 1536,
                k: 1928,
                n: 1536,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 1536,
                k: 1928,
                n: 2304,
            },
            B,
            C,
        ),
        (
            InferenceShape {
                m: 2048,
                k: 384,
                n: 1928,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 2048,
                k: 768,
                n: 1928,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 2048,
                k: 1928,
                n: 2304,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 3072,
                k: 1928,
                n: 1536,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 4096,
                k: 768,
                n: 2304,
            },
            B,
            A,
        ),
        (
            InferenceShape {
                m: 4621,
                k: 1928,
                n: 2304,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 1024,
                k: 1928,
                n: 1536,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 1024,
                k: 1928,
                n: 2304,
            },
            B,
            C,
        ),
        (
            InferenceShape {
                m: 2048,
                k: 1928,
                n: 1536,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 3072,
                k: 1928,
                n: 2304,
            },
            B,
            C,
        ),
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    assert_eq!(
        device.multiprocessor_count(),
        170,
        "170-SM qualification required"
    );
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (shape, incumbent, challenger) in cells {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
            let operands = InferenceFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let incumbent = InferenceTile::Sm120Half(incumbent);
            let challenger = InferenceTile::Sm120Half(challenger);
            for _ in 0..128 {
                inference_forward_with_tile(&ctx, operands, shape, incumbent)
                    .expect("incumbent warmup");
                inference_forward_with_tile(&ctx, operands, shape, challenger)
                    .expect("challenger warmup");
            }
            ctx.stream.synchronize().expect("paired warmup sync");
            let incumbent_iterations =
                fixed_tile_window_iterations(&ctx, operands, shape, incumbent);
            let challenger_iterations =
                fixed_tile_window_iterations(&ctx, operands, shape, challenger);
            let mut ratios = Vec::with_capacity(101);
            for round in 0..101 {
                let (challenger_us, incumbent_us) = if round & 1 == 0 {
                    (
                        fixed_tile_window_us(
                            &ctx,
                            operands,
                            shape,
                            challenger,
                            challenger_iterations,
                        ),
                        fixed_tile_window_us(
                            &ctx,
                            operands,
                            shape,
                            incumbent,
                            incumbent_iterations,
                        ),
                    )
                } else {
                    let incumbent_us = fixed_tile_window_us(
                        &ctx,
                        operands,
                        shape,
                        incumbent,
                        incumbent_iterations,
                    );
                    let challenger_us = fixed_tile_window_us(
                        &ctx,
                        operands,
                        shape,
                        challenger,
                        challenger_iterations,
                    );
                    (challenger_us, incumbent_us)
                };
                ratios.push(challenger_us / incumbent_us);
            }
            ratios.sort_by(f64::total_cmp);
            println!(
                "dtype={dtype:?} m={} k={} n={} incumbent={incumbent:?} challenger={challenger:?} incumbent_iterations={incumbent_iterations} challenger_iterations={challenger_iterations} ratio_p05={:.6} p50={:.6} p95={:.6}",
                shape.m,
                shape.k,
                shape.n,
                percentile(&ratios, 0.05),
                percentile(&ratios, 0.50),
                percentile(&ratios, 0.95),
            );
        }
    }
}

#[test]
#[ignore = "requires a quiet 170-SM CC12 CUDA device and screens every SM120 half tile"]
fn fixed_sm120_half_remaining_gap_all_tile_screen() {
    let shapes = [
        InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 512,
            k: 1928,
            n: 2304,
        },
        InferenceShape {
            m: 1024,
            k: 1928,
            n: 1928,
        },
        InferenceShape {
            m: 1024,
            k: 1928,
            n: 2304,
        },
        InferenceShape {
            m: 1536,
            k: 1032,
            n: 1536,
        },
        InferenceShape {
            m: 1536,
            k: 1928,
            n: 1536,
        },
        InferenceShape {
            m: 1536,
            k: 1928,
            n: 2304,
        },
        InferenceShape {
            m: 2048,
            k: 1928,
            n: 1536,
        },
        InferenceShape {
            m: 2048,
            k: 1928,
            n: 2304,
        },
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 1928,
        },
        InferenceShape {
            m: 1536,
            k: 384,
            n: 1536,
        },
        InferenceShape {
            m: 1536,
            k: 768,
            n: 1536,
        },
        InferenceShape {
            m: 3072,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 4621,
            k: 768,
            n: 1536,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    let candidate_filter = std::env::var("MAMBA_FIXED_HALF_TILE_CANDIDATE").ok();
    if let Some(filter) = candidate_filter.as_deref() {
        assert!(
            InferenceSm120HalfTile::ALL
                .iter()
                .any(|tile| format!("{tile:?}") == filter),
            "unknown Fixed SM120 half candidate filter: {filter}"
        );
    }

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let incumbent_output =
                DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("incumbent output");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa170_5010))
                .expect("A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb170_5010))
                .expect("B upload");
            let incumbent_operands = InferenceFwdOperands {
                c: typed(&incumbent_output, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let selected = inference_forward(
                &ctx,
                incumbent_operands.c,
                incumbent_operands.x,
                incumbent_operands.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("production AUTO launch");
            let InferenceTile::Sm120Half(incumbent) = selected else {
                panic!("production AUTO selected {selected:?} for {shape:?}");
            };
            ctx.stream.synchronize().expect("incumbent result sync");
            let expected = f32_bits(&ctx, &incumbent_output, shape.m * shape.n);
            let incumbent = InferenceTile::Sm120Half(incumbent);

            for candidate in InferenceSm120HalfTile::ALL {
                if candidate_filter
                    .as_deref()
                    .is_some_and(|filter| format!("{candidate:?}") != filter)
                {
                    continue;
                }
                let candidate = InferenceTile::Sm120Half(candidate);
                if candidate == incumbent {
                    continue;
                }
                let candidate_output = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype)
                    .expect("candidate output");
                let candidate_operands = InferenceFwdOperands {
                    c: typed(&candidate_output, dtype),
                    ..incumbent_operands
                };
                inference_forward_with_tile(&ctx, candidate_operands, shape, candidate)
                    .expect("candidate bit-gate launch");
                ctx.stream.synchronize().expect("candidate bit-gate sync");
                assert_eq!(
                    f32_bits(&ctx, &candidate_output, shape.m * shape.n),
                    expected,
                    "candidate bits for {dtype:?} {shape:?} {candidate:?}"
                );
                for _ in 0..128 {
                    inference_forward_with_tile(&ctx, incumbent_operands, shape, incumbent)
                        .expect("incumbent warmup");
                    inference_forward_with_tile(&ctx, candidate_operands, shape, candidate)
                        .expect("candidate warmup");
                }
                ctx.stream.synchronize().expect("paired warmup sync");
                let incumbent_iterations =
                    fixed_tile_window_iterations(&ctx, incumbent_operands, shape, incumbent);
                let candidate_iterations =
                    fixed_tile_window_iterations(&ctx, candidate_operands, shape, candidate);
                let mut candidate_first = Vec::with_capacity(101);
                let mut incumbent_first = Vec::with_capacity(101);
                for order in 0..2 {
                    let ratios = if order == 0 {
                        &mut candidate_first
                    } else {
                        &mut incumbent_first
                    };
                    for _ in 0..101 {
                        let (candidate_us, incumbent_us) = if order == 0 {
                            (
                                fixed_tile_window_us(
                                    &ctx,
                                    candidate_operands,
                                    shape,
                                    candidate,
                                    candidate_iterations,
                                ),
                                fixed_tile_window_us(
                                    &ctx,
                                    incumbent_operands,
                                    shape,
                                    incumbent,
                                    incumbent_iterations,
                                ),
                            )
                        } else {
                            let incumbent_us = fixed_tile_window_us(
                                &ctx,
                                incumbent_operands,
                                shape,
                                incumbent,
                                incumbent_iterations,
                            );
                            let candidate_us = fixed_tile_window_us(
                                &ctx,
                                candidate_operands,
                                shape,
                                candidate,
                                candidate_iterations,
                            );
                            (candidate_us, incumbent_us)
                        };
                        ratios.push(candidate_us / incumbent_us);
                    }
                    ratios.sort_by(f64::total_cmp);
                }
                println!(
                    concat!(
                        "dtype={:?} m={} k={} n={} incumbent={:?} ",
                        "candidate={:?} candidate_first_p50={:.6} ",
                        "candidate_first_p95={:.6} incumbent_first_p50={:.6} ",
                        "incumbent_first_p95={:.6}"
                    ),
                    dtype,
                    shape.m,
                    shape.k,
                    shape.n,
                    incumbent,
                    candidate,
                    percentile(&candidate_first, 0.50),
                    percentile(&candidate_first, 0.95),
                    percentile(&incumbent_first, 0.50),
                    percentile(&incumbent_first, 0.95),
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet 170-SM CC12 CUDA device and emits paired selector boundary evidence"]
fn fixed_sm120_half_selector_paired_boundaries() {
    use InferenceSm120HalfTile::M128N128Bk32S2 as B;
    use InferenceSm120HalfTile::{M64N64Bk64S2 as C, M128N64Bk32S3 as A};

    let cells = [
        (
            InferenceShape {
                m: 1024,
                k: 1928,
                n: 1928,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 1536,
                k: 1928,
                n: 1928,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 2048,
                k: 1928,
                n: 1928,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 3072,
                k: 1928,
                n: 1928,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 4621,
                k: 1928,
                n: 1928,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 1536,
                k: 1032,
                n: 1536,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 2048,
                k: 1032,
                n: 1536,
            },
            C,
            B,
        ),
        (
            InferenceShape {
                m: 1536,
                k: 384,
                n: 1928,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 2048,
                k: 384,
                n: 1536,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 2048,
                k: 768,
                n: 1536,
            },
            C,
            A,
        ),
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    assert_eq!(
        device.multiprocessor_count(),
        170,
        "170-SM qualification required"
    );
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (shape, incumbent, challenger) in cells {
            for challenger_first in [true, false] {
                run_fixed_sm120_half_selector_boundary_cohort(
                    &ctx,
                    dtype,
                    shape,
                    incumbent,
                    challenger,
                    challenger_first,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet 170-SM CC12 CUDA device and emits paired shallow selector boundary evidence"]
fn fixed_sm120_half_selector_paired_shallow_boundaries() {
    use InferenceSm120HalfTile::M128N128Bk32S2 as B;
    use InferenceSm120HalfTile::{M64N64Bk64S2 as C, M128N64Bk32S3 as A};

    let cells = [
        (
            InferenceShape {
                m: 3072,
                k: 384,
                n: 1928,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 4096,
                k: 384,
                n: 1536,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 4096,
                k: 768,
                n: 1536,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 4096,
                k: 384,
                n: 1928,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 4096,
                k: 768,
                n: 1928,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 4621,
                k: 384,
                n: 1536,
            },
            A,
            C,
        ),
        (
            InferenceShape {
                m: 4621,
                k: 768,
                n: 1536,
            },
            A,
            C,
        ),
        (
            InferenceShape {
                m: 3072,
                k: 768,
                n: 2304,
            },
            C,
            A,
        ),
        (
            InferenceShape {
                m: 4621,
                k: 768,
                n: 2304,
            },
            B,
            A,
        ),
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    assert!(matches!(device.compute_capability, (12, 0) | (12, 1)));
    assert_eq!(
        device.multiprocessor_count(),
        170,
        "170-SM qualification required"
    );
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (shape, incumbent, challenger) in cells {
            for challenger_first in [true, false] {
                run_fixed_sm120_half_selector_boundary_cohort(
                    &ctx,
                    dtype,
                    shape,
                    incumbent,
                    challenger,
                    challenger_first,
                );
            }
        }
    }
}

fn run_fixed_sm120_half_selector_boundary_cohort(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    shape: InferenceShape,
    incumbent: InferenceSm120HalfTile,
    challenger: InferenceSm120HalfTile,
    challenger_first: bool,
) {
    let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
    let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
    let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
    let operands = InferenceFwdOperands {
        c: typed(&c, dtype),
        x: typed(&a, dtype),
        w: typed(&b, dtype),
        bias_ptr: None,
    };
    let incumbent = InferenceTile::Sm120Half(incumbent);
    let challenger = InferenceTile::Sm120Half(challenger);
    for _ in 0..128 {
        inference_forward_with_tile(ctx, operands, shape, incumbent).expect("incumbent warmup");
        inference_forward_with_tile(ctx, operands, shape, challenger).expect("challenger warmup");
    }
    ctx.stream.synchronize().expect("paired warmup sync");
    let incumbent_iterations = fixed_tile_window_iterations(ctx, operands, shape, incumbent);
    let challenger_iterations = fixed_tile_window_iterations(ctx, operands, shape, challenger);
    let mut incumbent_us = Vec::with_capacity(101);
    let mut challenger_us = Vec::with_capacity(101);
    let mut ratios = Vec::with_capacity(101);
    for _ in 0..101 {
        let (challenger_elapsed_us, incumbent_elapsed_us) = if challenger_first {
            (
                fixed_tile_window_us(ctx, operands, shape, challenger, challenger_iterations),
                fixed_tile_window_us(ctx, operands, shape, incumbent, incumbent_iterations),
            )
        } else {
            let incumbent_elapsed_us =
                fixed_tile_window_us(ctx, operands, shape, incumbent, incumbent_iterations);
            let challenger_elapsed_us =
                fixed_tile_window_us(ctx, operands, shape, challenger, challenger_iterations);
            (challenger_elapsed_us, incumbent_elapsed_us)
        };
        ratios.push(challenger_elapsed_us / incumbent_elapsed_us);
        challenger_us.push(challenger_elapsed_us);
        incumbent_us.push(incumbent_elapsed_us);
    }
    challenger_us.sort_by(f64::total_cmp);
    incumbent_us.sort_by(f64::total_cmp);
    ratios.sort_by(f64::total_cmp);
    let order = if challenger_first {
        "challenger_then_incumbent"
    } else {
        "incumbent_then_challenger"
    };
    println!(
        "dtype={dtype:?} m={} k={} n={} incumbent={incumbent:?} challenger={challenger:?} order={order} pairs=101 incumbent_iterations={incumbent_iterations} challenger_iterations={challenger_iterations} challenger_p50_us={:.6} challenger_p95_us={:.6} incumbent_p50_us={:.6} incumbent_p95_us={:.6} ratio_p50={:.6} ratio_p95={:.6}",
        shape.m,
        shape.k,
        shape.n,
        percentile(&challenger_us, 0.50),
        percentile(&challenger_us, 0.95),
        percentile(&incumbent_us, 0.50),
        percentile(&incumbent_us, 0.95),
        percentile(&ratios, 0.50),
        percentile(&ratios, 0.95),
    );
}

const FIXED_SM120_HALF_EXACT_JSONL_ENV: &str = "GEMM_BI_FIXED_HALF_EXACT_JSONL";

#[derive(Clone, Copy)]
struct FixedSm120HalfRequalificationCell {
    label: &'static str,
    dtype: WeightDtype,
    shape: InferenceShape,
    has_bias: bool,
    candidate: InferenceSm120HalfTile,
    incumbent: InferenceSm120HalfTile,
}

fn sha256_file(path: &std::path::Path) -> String {
    let bytes =
        std::fs::read(path).unwrap_or_else(|error| panic!("read SHA-256 input {path:?}: {error}"));
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fixed_sm120_half_emit_requalification_cohort(
    ctx: &GpuCtx,
    writer: &mut BufWriter<File>,
    cell: FixedSm120HalfRequalificationCell,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
    candidate_first: bool,
) -> (f64, f64) {
    let a = DtypedBuf::zeros(&ctx.stream, cell.shape.m * cell.shape.k, cell.dtype)
        .expect("paired A allocation");
    let b = DtypedBuf::zeros(&ctx.stream, cell.shape.k * cell.shape.n, cell.dtype)
        .expect("paired B allocation");
    let output = DtypedBuf::zeros(&ctx.stream, cell.shape.m * cell.shape.n, cell.dtype)
        .expect("paired output allocation");
    a.upload_f32(
        &ctx.stream,
        &synth(
            cell.shape.m * cell.shape.k,
            0xa132_0001 ^ cell.shape.m as u64,
        ),
    )
    .expect("paired A upload");
    b.upload_f32(
        &ctx.stream,
        &synth(
            cell.shape.k * cell.shape.n,
            0xb132_0002 ^ cell.shape.n as u64,
        ),
    )
    .expect("paired B upload");
    let operands = InferenceFwdOperands {
        c: typed(&output, cell.dtype),
        x: typed(&a, cell.dtype),
        w: typed(&b, cell.dtype),
        bias_ptr,
    };
    let candidate = InferenceTile::Sm120Half(cell.candidate);
    let incumbent = InferenceTile::Sm120Half(cell.incumbent);
    for _ in 0..128 {
        inference_forward_with_tile(ctx, operands, cell.shape, candidate)
            .expect("candidate warmup");
        inference_forward_with_tile(ctx, operands, cell.shape, incumbent)
            .expect("incumbent warmup");
    }
    ctx.stream
        .synchronize()
        .expect("paired warmup synchronization");
    let candidate_iterations = fixed_tile_window_iterations(ctx, operands, cell.shape, candidate);
    let incumbent_iterations = fixed_tile_window_iterations(ctx, operands, cell.shape, incumbent);
    let order = if candidate_first {
        "candidate_then_incumbent"
    } else {
        "incumbent_then_candidate"
    };
    let bias = if bias_ptr.is_some() {
        "synthesized"
    } else {
        "none"
    };
    fixed_sm120_tf32_bd_environment_preflight(&format!(
        "HALF exact cell={} dtype={} m={} k={} n={} bias={bias} order={order}",
        cell.label,
        cell.dtype.as_str(),
        cell.shape.m,
        cell.shape.k,
        cell.shape.n,
    ))
    .unwrap_or_else(|error| panic!("HALF exact timed-set preflight failed: {error}"));
    let mut ratios = Vec::with_capacity(101);
    for pair_index in 0..101 {
        let (candidate_us, incumbent_us) = if candidate_first {
            (
                fixed_tile_window_us(ctx, operands, cell.shape, candidate, candidate_iterations),
                fixed_tile_window_us(ctx, operands, cell.shape, incumbent, incumbent_iterations),
            )
        } else {
            let incumbent_us =
                fixed_tile_window_us(ctx, operands, cell.shape, incumbent, incumbent_iterations);
            let candidate_us =
                fixed_tile_window_us(ctx, operands, cell.shape, candidate, candidate_iterations);
            (candidate_us, incumbent_us)
        };
        let ratio = candidate_us / incumbent_us;
        assert!(ratio.is_finite() && ratio > 0.0, "valid paired ratio");
        ratios.push(ratio);
        writeln!(
            writer,
            "{{\"schema\":\"MambaBiFixedSm120HalfExactPairedV1\",\"record_type\":\"window\",\"cell\":\"{}\",\"dtype\":\"{}\",\"m\":{},\"k\":{},\"n\":{},\"bias\":\"{bias}\",\"order\":\"{order}\",\"pair_index\":{pair_index},\"candidate_tile\":\"{:?}\",\"incumbent_tile\":\"{:?}\",\"candidate_iterations\":{candidate_iterations},\"incumbent_iterations\":{incumbent_iterations},\"candidate_us\":{candidate_us:.9},\"incumbent_us\":{incumbent_us:.9},\"ratio\":{ratio:.9}}}",
            cell.label,
            cell.dtype.as_str(),
            cell.shape.m,
            cell.shape.k,
            cell.shape.n,
            cell.candidate,
            cell.incumbent,
        )
        .expect("write paired window");
    }
    ratios.sort_by(f64::total_cmp);
    let ratio_p50 = percentile(&ratios, 0.50);
    let ratio_p95 = percentile(&ratios, 0.95);
    let passed = ratio_p50 < 1.0 && ratio_p95 < 1.0;
    writeln!(
        writer,
        "{{\"schema\":\"MambaBiFixedSm120HalfExactPairedV1\",\"record_type\":\"summary\",\"cell\":\"{}\",\"dtype\":\"{}\",\"m\":{},\"k\":{},\"n\":{},\"bias\":\"{bias}\",\"order\":\"{order}\",\"windows\":101,\"warmups\":128,\"pilot_iterations\":16,\"target_window_ms\":5.0,\"candidate_tile\":\"{:?}\",\"incumbent_tile\":\"{:?}\",\"candidate_iterations\":{candidate_iterations},\"incumbent_iterations\":{incumbent_iterations},\"ratio_p50\":{ratio_p50:.9},\"ratio_p95\":{ratio_p95:.9},\"passed\":{passed}}}",
        cell.label,
        cell.dtype.as_str(),
        cell.shape.m,
        cell.shape.k,
        cell.shape.n,
        cell.candidate,
        cell.incumbent,
    )
    .expect("write paired summary");
    writer.flush().expect("flush paired summary");
    println!(
        "cell={} dtype={} bias={bias} order={order} ratio_p50={ratio_p50:.9} ratio_p95={ratio_p95:.9} passed={passed}",
        cell.label,
        cell.dtype.as_str(),
    );
    (ratio_p50, ratio_p95)
}

#[test]
#[ignore = "requires a quiet RTX5090 with NVRTC 13.2 and a create-new JSONL sink"]
fn fixed_sm120_half_legacy_overlay_requalification() {
    use InferenceSm120HalfTile::{M64N64Bk64S2 as C, M128N64Bk32S3 as A};

    let path = std::env::var_os(FIXED_SM120_HALF_EXACT_JSONL_ENV)
        .expect("GEMM_BI_FIXED_HALF_EXACT_JSONL must name a new JSONL sink");
    assert!(!path.is_empty(), "JSONL sink path must not be empty");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap_or_else(|error| panic!("create-new HALF exact JSONL sink {path:?}: {error}"));
    let mut writer = BufWriter::new(file);

    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let compiler = ctx.kernels.compiler_identity();
    assert_eq!(compiler.nvrtc_version, (13, 2));
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);

    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let rust_source_sha256 = sha256_file(&manifest.join("src/mamba_ssm/gpu/gemm_bi_inference.rs"));
    let cuda_source_sha256 = sha256_file(&manifest.join("kernels/gemm_bi_inference/sm120/tma.cu"));
    let executable_sha256 =
        sha256_file(&std::env::current_exe().expect("current qualification executable path"));
    let fixed_module_source_sha256 = digest_hex(&compiler.source_digest);
    let fixed_module_invocation_sha256 = digest_hex(&compiler.invocation_digest);
    let device_sha256 = sha256_bytes(format!("{:?}", device.identity()).as_bytes());
    writeln!(
        writer,
        "{{\"schema\":\"MambaBiFixedSm120HalfExactPairedV1\",\"record_type\":\"metadata\",\"device_cc\":\"12.0\",\"sm_count\":170,\"device_sha256\":\"{device_sha256}\",\"nvrtc_major\":{},\"nvrtc_minor\":{},\"compiler_target\":\"{:?}\",\"fixed_module_source_sha256\":\"{fixed_module_source_sha256}\",\"fixed_module_invocation_sha256\":\"{fixed_module_invocation_sha256}\",\"rust_source_sha256\":\"{rust_source_sha256}\",\"cuda_source_sha256\":\"{cuda_source_sha256}\",\"executable_sha256\":\"{executable_sha256}\",\"tuning_table_revision\":{TUNING_TABLE_REVISION}}}",
        compiler.nvrtc_version.0,
        compiler.nvrtc_version.1,
        compiler.target,
    )
    .expect("write paired metadata");

    let cells = [
        FixedSm120HalfRequalificationCell {
            label: "A1_f16_none",
            dtype: WeightDtype::F16,
            shape: InferenceShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            has_bias: false,
            candidate: A,
            incumbent: C,
        },
        FixedSm120HalfRequalificationCell {
            label: "A1_f16_bias",
            dtype: WeightDtype::F16,
            shape: InferenceShape {
                m: 3072,
                k: 768,
                n: 1928,
            },
            has_bias: true,
            candidate: A,
            incumbent: C,
        },
        FixedSm120HalfRequalificationCell {
            label: "A3_f16_none",
            dtype: WeightDtype::F16,
            shape: InferenceShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            has_bias: false,
            candidate: A,
            incumbent: C,
        },
        FixedSm120HalfRequalificationCell {
            label: "A3_f16_bias",
            dtype: WeightDtype::F16,
            shape: InferenceShape {
                m: 1536,
                k: 768,
                n: 1928,
            },
            has_bias: true,
            candidate: A,
            incumbent: C,
        },
    ];
    let bias =
        DtypedBuf::zeros(&ctx.stream, 1928, WeightDtype::F32).expect("paired bias allocation");
    bias.upload_f32(&ctx.stream, &synth(1928, 0xb1a5_0132))
        .expect("paired bias upload");
    let mut all_candidate_contexts_passed = true;
    let mut context_passed = Vec::with_capacity(cells.len());
    for cell in cells {
        let bias_ptr = cell.has_bias.then(|| bias.cached_ptr());
        let mut cell_passed = true;
        for candidate_first in [true, false] {
            let (p50, p95) = fixed_sm120_half_emit_requalification_cohort(
                &ctx,
                &mut writer,
                cell,
                bias_ptr,
                candidate_first,
            );
            cell_passed &= p50 < 1.0 && p95 < 1.0;
        }
        all_candidate_contexts_passed &= cell_passed;
        context_passed.push((cell.label, cell_passed));
    }
    let contexts_json = context_passed
        .iter()
        .map(|(label, passed)| format!(r#"{{"cell":"{label}","passed":{passed}}}"#))
        .collect::<Vec<_>>()
        .join(",");
    writeln!(
        writer,
        "{{\"schema\":\"MambaBiFixedSm120HalfExactPairedV1\",\"record_type\":\"suite_decision\",\"all_candidate_contexts_passed\":{all_candidate_contexts_passed},\"production_retention_effect\":false,\"contexts\":[{contexts_json}]}}",
    )
    .expect("write paired suite decision");
    writer.flush().expect("flush paired JSONL");
}

fn run_fixed_sm120_half_selector_cell(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    shape: InferenceShape,
    candidates: [InferenceSm120HalfTile; 3],
) {
    let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
    let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
    let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
    let operands = InferenceFwdOperands {
        c: typed(&c, dtype),
        x: typed(&a, dtype),
        w: typed(&b, dtype),
        bias_ptr: None,
    };
    let mut portable_tile = InferenceTile::Legacy;
    let portable_us = average_us(ctx, || {
        portable_tile = inference_forward(
            ctx,
            operands.c,
            operands.x,
            operands.w,
            None,
            (shape.m, shape.k, shape.n),
        )
        .expect("portable auto launch");
    });
    for candidate in candidates {
        let elapsed_us = average_us(ctx, || {
            inference_forward_with_tile(ctx, operands, shape, InferenceTile::Sm120Half(candidate))
                .unwrap_or_else(|error| panic!("forced {candidate:?}: {error}"));
        });
        println!(
            "m={} k={} n={} portable={portable_tile:?} portable_us={portable_us:.3} tile={candidate:?} elapsed_us={elapsed_us:.3} over_portable={:.5}",
            shape.m,
            shape.k,
            shape.n,
            elapsed_us / portable_us,
        );
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits mixed-output data"]
fn half_to_f32_hot_shapes_smoke() {
    let shapes = [
        InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let legacy = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("legacy allocation");
            let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("candidate allocation");
            let cublas = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
                .expect("cuBLAS allocation");
            let legacy_ops = InferenceFwdOperands {
                c: typed(&legacy, WeightDtype::F32),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            let candidate_ops = InferenceFwdOperands {
                c: typed(&candidate, WeightDtype::F32),
                ..legacy_ops
            };
            let cublas_ops = InferenceFwdOperands {
                c: typed(&cublas, WeightDtype::F32),
                ..legacy_ops
            };
            let legacy_us = average_us(&ctx, || {
                inference_forward_with_tile(&ctx, legacy_ops, shape, InferenceTile::Legacy)
                    .expect("legacy mixed-output launch");
            });
            let forced_us = [
                InferenceTile::Tc16,
                InferenceTile::Tc64,
                InferenceTile::Tc128,
            ]
            .map(|tile| {
                average_us(&ctx, || {
                    inference_forward_with_tile(&ctx, candidate_ops, shape, tile)
                        .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                })
            });
            let mut selected = InferenceTile::Legacy;
            let candidate_us = average_us(&ctx, || {
                selected = inference_forward(
                    &ctx,
                    candidate_ops.c,
                    candidate_ops.x,
                    candidate_ops.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("auto mixed-output launch");
            });
            assert!(matches!(
                selected,
                InferenceTile::Tc16 | InferenceTile::Tc64 | InferenceTile::Tc128
            ));
            ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
            let cublas_us = average_us(&ctx, || {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    cublas_ops.c,
                    cublas_ops.x,
                    cublas_ops.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("fast cuBLAS mixed-output launch");
            });
            ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
            println!(
                "dtype={dtype:?} m={} k={} n={} legacy_us={legacy_us:.3} tc16_us={:.3} tc64_us={:.3} tc128_us={:.3} selected={selected:?} auto_us={candidate_us:.3} cublas_us={cublas_us:.3} auto_over_legacy={:.5} auto_over_cublas={:.5}",
                shape.m,
                shape.k,
                shape.n,
                forced_us[0],
                forced_us[1],
                forced_us[2],
                candidate_us / legacy_us,
                candidate_us / cublas_us,
            );
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits F32 candidate data"]
fn f32_s2_hot_shapes_smoke() {
    let shapes = [
        InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for shape in shapes {
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::F32)
            .expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::F32)
            .expect("B allocation");
        let baseline = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("baseline allocation");
        let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("candidate allocation");
        let cublas = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("cuBLAS allocation");
        let base_ops = InferenceFwdOperands {
            c: typed(&baseline, WeightDtype::F32),
            x: typed(&a, WeightDtype::F32),
            w: typed(&b, WeightDtype::F32),
            bias_ptr: None,
        };
        let candidate_ops = InferenceFwdOperands {
            c: typed(&candidate, WeightDtype::F32),
            ..base_ops
        };
        let cublas_ops = InferenceFwdOperands {
            c: typed(&cublas, WeightDtype::F32),
            ..base_ops
        };
        let baseline_us = average_us(&ctx, || {
            inference_forward_f32_legacy_baseline(&ctx, base_ops, shape)
                .expect("legacy Fixed launch");
        });
        let candidate_us = average_us(&ctx, || {
            inference_forward(
                &ctx,
                candidate_ops.c,
                candidate_ops.x,
                candidate_ops.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("production S2 launch");
        });
        ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
        let cublas_us = average_us(&ctx, || {
            gpu_gemm_typed_forward_raw(
                &ctx,
                cublas_ops.c,
                cublas_ops.x,
                cublas_ops.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("fast cuBLAS launch");
        });
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        println!(
            "m={} k={} n={} legacy_us={baseline_us:.3} s2_us={candidate_us:.3} cublas_tf32_us={cublas_us:.3} s2_over_legacy={:.5} s2_over_cublas={:.5}",
            shape.m,
            shape.k,
            shape.n,
            candidate_us / baseline_us,
            candidate_us / cublas_us,
        );
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn forced_fixed_portable_tiles_execute() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    let shape = InferenceShape {
        m: 128,
        k: 128,
        n: 128,
    };
    let a =
        DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, WeightDtype::Bf16).expect("A allocation");
    let b =
        DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, WeightDtype::Bf16).expect("B allocation");
    let c =
        DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::Bf16).expect("C allocation");
    let operands = InferenceFwdOperands {
        c: TypedPtr {
            ptr: c.cached_ptr(),
            dtype: WeightDtype::Bf16,
        },
        x: TypedPtr {
            ptr: a.cached_ptr(),
            dtype: WeightDtype::Bf16,
        },
        w: TypedPtr {
            ptr: b.cached_ptr(),
            dtype: WeightDtype::Bf16,
        },
        bias_ptr: None,
    };
    for tile in [
        InferenceTile::Tc16,
        InferenceTile::Tc64,
        InferenceTile::Tc128,
        InferenceTile::TcWn64,
    ] {
        inference_forward_with_tile(&ctx, operands, shape, tile)
            .unwrap_or_else(|error| panic!("forced {tile:?} launch: {error}"));
    }
    ctx.stream.synchronize().expect("forced launch sync");
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits performance data"]
fn fixed_hot_shapes_smoke() {
    let shapes = [
        InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
            let operands = InferenceFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };

            ctx.set_gemm_mode(GemmMode::CublasPedantic).unwrap();
            ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
            ctx.set_bi_gemm_family(BiGemmFamily::Inference);
            let mut selected = InferenceTile::Legacy;
            let fixed_us = average_us(&ctx, || {
                selected = inference_forward(
                    &ctx,
                    operands.c,
                    operands.x,
                    operands.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("Fixed launch");
            });

            ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
            let cublas_us = average_us(&ctx, || {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    operands.c,
                    operands.x,
                    operands.w,
                    None,
                    (shape.m, shape.k, shape.n),
                )
                .expect("fast cuBLAS launch");
            });
            println!(
                "dtype={dtype:?} m={} k={} n={} tile={selected:?} fixed_us={fixed_us:.3} cublas_fast_us={cublas_us:.3} fixed_over_cublas={:.5}",
                shape.m,
                shape.k,
                shape.n,
                fixed_us / cublas_us,
            );
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits forced-route performance data"]
fn fixed_forced_hot_shapes_smoke() {
    let shapes = [
        InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];
    let tiles = [
        InferenceTile::Tc16,
        InferenceTile::Tc64,
        InferenceTile::Tc128,
        InferenceTile::TcW64,
        InferenceTile::TcWn64,
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for shape in shapes {
        let dtype = WeightDtype::Bf16;
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
        let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
        let operands = InferenceFwdOperands {
            c: typed(&c, dtype),
            x: typed(&a, dtype),
            w: typed(&b, dtype),
            bias_ptr: None,
        };
        for tile in tiles {
            let elapsed_us = average_us(&ctx, || {
                inference_forward_with_tile(&ctx, operands, shape, tile)
                    .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
            });
            println!(
                "m={} k={} n={} tile={tile:?} elapsed_us={elapsed_us:.3}",
                shape.m, shape.k, shape.n,
            );
        }
    }
}

#[test]
#[ignore = "requires an SM120 CUDA device and emits W64 selector data"]
fn fixed_w64_deep_k_grid_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for m in [1024usize, 2048, 4621] {
            for (k, n) in [(1024usize, 384usize), (1536, 768), (1928, 384), (2304, 768)] {
                let shape = InferenceShape { m, k, n };
                let a =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
                let b =
                    DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
                let c =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
                let operands = InferenceFwdOperands {
                    c: typed(&c, dtype),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr: None,
                };
                let square_us = average_us(&ctx, || {
                    inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc128)
                        .expect("Tc128 launch");
                });
                let reuse_us = average_us(&ctx, || {
                    inference_forward_with_tile(&ctx, operands, shape, InferenceTile::TcW64)
                        .expect("TcW64 launch");
                });
                println!(
                    "dtype={dtype:?} m={m} k={k} n={n} tc128_us={square_us:.3} tcw64_us={reuse_us:.3} tcw64_over_tc128={:.5}",
                    reuse_us / square_us,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn fixed_portable_resource_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let f32_registers = ctx
        .kernels
        .gemm_bi_f32_f32_s2
        .num_regs()
        .expect("F32 S2 register count");
    let f32_local_bytes = ctx
        .kernels
        .gemm_bi_f32_f32_s2
        .local_size_bytes()
        .expect("F32 S2 local size");
    println!("dtype=F32 tile=S2 registers={f32_registers} local_bytes={f32_local_bytes}");
    assert_eq!(f32_local_bytes, 0, "F32 S2 spills to local memory");
    for (tile, function) in [
        (
            InferenceTile::Tf32M128S2,
            &ctx.kernels.gemm_bi_nn_tf32.m128n64_s2,
        ),
        (
            InferenceTile::Tf32M128S3,
            &ctx.kernels.gemm_bi_nn_tf32.m128n64_s3,
        ),
        (
            InferenceTile::Tf32M64S2,
            &ctx.kernels.gemm_bi_nn_tf32.m64n64_s2,
        ),
        (
            InferenceTile::Tf32M64S3,
            &ctx.kernels.gemm_bi_nn_tf32.m64n64_s3,
        ),
        (
            InferenceTile::Tf32M16S4,
            &ctx.kernels.gemm_bi_nn_tf32.m16n32_s4,
        ),
    ] {
        let registers = function.num_regs().expect("TF32 register count");
        let local_bytes = function.local_size_bytes().expect("TF32 local size");
        println!("dtype=TF32 tile={tile:?} registers={registers} local_bytes={local_bytes}");
        assert_eq!(local_bytes, 0, "TF32 {tile:?} spills to local memory");
    }
    let m64 = &ctx.kernels.gemm_bi_nn_tf32.m64n64_s2;
    let m64_occupancy = m64
        .occupancy_max_active_blocks_per_multiprocessor(128, 32_768, None)
        .expect("TF32 M64 occupancy");
    println!("dtype=TF32 tile=Tf32M64S2 occupancy_blocks_per_sm={m64_occupancy}");
    if ctx
        .stream
        .context()
        .compute_capability()
        .expect("CUDA compute capability")
        == (8, 9)
    {
        assert_eq!(m64_occupancy, 3, "SM89 TF32 M64 residency changed");
    }
    if let Some(kernels) = &ctx.kernels.gemm_bi_nn_tf32_sm120 {
        for (tile, function) in [
            (InferenceTile::Tf32Sm120M128S2, &kernels.m128n64_s2),
            (InferenceTile::Tf32Sm120M128S3, &kernels.m128n64_s3),
            (
                InferenceTile::Tf32Sm120M64S2ProducerWarp,
                &kernels.m64n64_s2_producer_warp,
            ),
            (InferenceTile::Tf32Sm120M64N128S2, &kernels.m64n128_s2),
            (InferenceTile::Tf32Sm120M64N128S3, &kernels.m64n128_s3),
            (InferenceTile::Tf32Sm120M64S2, &kernels.m64n64_s2),
        ] {
            let registers = function.num_regs().expect("SM120 TF32 register count");
            let local_bytes = function.local_size_bytes().expect("SM120 TF32 local size");
            println!("dtype=TF32 tile={tile:?} registers={registers} local_bytes={local_bytes}");
            assert_eq!(local_bytes, 0, "SM120 TF32 {tile:?} spills to local memory");
        }
        let m64 = &kernels.m64n64_s2;
        let carveout = m64
            .get_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            )
            .expect("SM120 TF32 M64 carveout");
        let occupancy = m64
            .occupancy_max_active_blocks_per_multiprocessor(128, 32_896, None)
            .expect("SM120 TF32 M64 occupancy");
        println!(
            "dtype=TF32 tile=Tf32Sm120M64S2 preferred_carveout={carveout} occupancy_blocks_per_sm={occupancy}"
        );
        let producer_occupancy = kernels
            .m64n64_s2_producer_warp
            .occupancy_max_active_blocks_per_multiprocessor(160, 32_896, None)
            .expect("SM120 TF32 producer-warp occupancy");
        println!(
            "dtype=TF32 tile=Tf32Sm120M64S2ProducerWarp occupancy_blocks_per_sm={producer_occupancy}"
        );
        assert!(
            producer_occupancy >= 3,
            "SM120 TF32 producer-warp candidate must preserve three resident CTAs per SM"
        );
    }
    for (output, kernels) in [
        ("half", ctx.kernels.gemm_bi_nn_half_sm120.as_ref()),
        ("f32", ctx.kernels.gemm_bi_nn_half_sm120_f32out.as_ref()),
    ] {
        if let Some(kernels) = kernels {
            for (tile, functions) in [
                (
                    InferenceSm120HalfTile::M64N64Bk64S2,
                    &kernels.m64n64_bk64_s2,
                ),
                (
                    InferenceSm120HalfTile::M64N128Bk64S2,
                    &kernels.m64n128_bk64_s2,
                ),
                (
                    InferenceSm120HalfTile::M128N64Bk32S3,
                    &kernels.m128n64_bk32_s3,
                ),
                (
                    InferenceSm120HalfTile::M128N128Bk32S2,
                    &kernels.m128n128_bk32_s2,
                ),
                (
                    InferenceSm120HalfTile::M128N128Bk32S3,
                    &kernels.m128n128_bk32_s3,
                ),
            ] {
                for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                    let function = functions.get(dtype);
                    let registers = function.num_regs().expect("SM120 half register count");
                    let local_bytes = function.local_size_bytes().expect("SM120 half local size");
                    println!(
                        "dtype={dtype:?} output={output} tile={tile:?} registers={registers} local_bytes={local_bytes}"
                    );
                    assert_eq!(
                        local_bytes, 0,
                        "SM120 half {dtype:?}->{output} {tile:?} spills"
                    );
                }
            }
        }
    }
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (tile, function) in [
            (
                InferenceTile::Tc16,
                ctx.kernels.gemm_bi_nn_tc16_typed.get(dtype),
            ),
            (
                InferenceTile::Tc64,
                ctx.kernels.gemm_bi_nn_tc64_typed.get(dtype),
            ),
            (
                InferenceTile::Tc128,
                ctx.kernels.gemm_bi_nn_tc128_typed.get(dtype),
            ),
            (
                InferenceTile::TcW64,
                ctx.kernels.gemm_bi_nn_tcw64_typed.get(dtype),
            ),
            (
                InferenceTile::TcWn64,
                ctx.kernels.gemm_bi_nn_tcwn64_typed.get(dtype),
            ),
        ] {
            let registers = function.num_regs().expect("register count");
            let local_bytes = function.local_size_bytes().expect("local size");
            let static_shared_bytes = function.shared_size_bytes().expect("shared size");
            println!(
                "dtype={dtype:?} tile={tile:?} registers={registers} local_bytes={local_bytes} static_shared_bytes={static_shared_bytes}"
            );
            assert_eq!(local_bytes, 0, "{dtype:?} {tile:?} spills to local memory");
            if tile == InferenceTile::Tc16 {
                assert_eq!(static_shared_bytes, 24_576);
                if ctx
                    .stream
                    .context()
                    .compute_capability()
                    .expect("CUDA compute capability")
                    == (8, 9)
                {
                    assert_eq!(
                        function
                            .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
                            .expect("Tc16 occupancy"),
                        4,
                        "SM89 Tc16 residency changed",
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an RTX5090 with the Fixed module loaded by NVRTC 13.2"]
fn fixed_sm120_half_loaded_resources_are_safe() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert_eq!(ctx.kernels.compiler_identity().nvrtc_version, (13, 2));
    let kernels = ctx
        .kernels
        .gemm_bi_nn_half_sm120
        .as_ref()
        .expect("Fixed SM120 half module");
    for (tile, functions, threads, dynamic_shared_bytes) in [
        (
            InferenceSm120HalfTile::M64N64Bk64S2,
            &kernels.m64n64_bk64_s2,
            128,
            32_896,
        ),
        (
            InferenceSm120HalfTile::M128N64Bk32S3,
            &kernels.m128n64_bk32_s3,
            256,
            36_992,
        ),
        (
            InferenceSm120HalfTile::M64N128Bk64S2,
            &kernels.m64n128_bk64_s2,
            256,
            49_280,
        ),
        (
            InferenceSm120HalfTile::M128N128Bk32S2,
            &kernels.m128n128_bk32_s2,
            256,
            32_896,
        ),
        (
            InferenceSm120HalfTile::M128N128Bk32S3,
            &kernels.m128n128_bk32_s3,
            256,
            49_280,
        ),
    ] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let function = functions.get(dtype);
            let registers = function.num_regs().expect("SM120 half register count");
            let local_bytes = function.local_size_bytes().expect("SM120 half local bytes");
            let static_shared_bytes = function
                .shared_size_bytes()
                .expect("SM120 half static shared bytes");
            let preferred_carveout = function
                .get_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                )
                .expect("SM120 half preferred shared-memory carveout");
            let max_dynamic_shared_bytes = function
                .get_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                )
                .expect("SM120 half maximum dynamic shared memory");
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(threads, dynamic_shared_bytes, None)
                .expect("SM120 half occupancy");
            println!(
                "{{\"schema\":\"MambaBiFixedSm120HalfResourceV1\",\"dtype\":\"{}\",\"tile\":\"{tile:?}\",\"registers\":{registers},\"local_bytes\":{local_bytes},\"static_shared_bytes\":{static_shared_bytes},\"threads\":{threads},\"dynamic_shared_bytes\":{dynamic_shared_bytes},\"max_dynamic_shared_bytes\":{max_dynamic_shared_bytes},\"preferred_shared_memory_carveout\":{preferred_carveout},\"occupancy_blocks_per_sm\":{occupancy}}}",
                dtype.as_str(),
            );
            assert!(
                registers <= 128,
                "{dtype:?} {tile:?} register safety ceiling"
            );
            assert_eq!(local_bytes, 0, "{dtype:?} {tile:?} local memory");
            assert_eq!(
                static_shared_bytes, 0,
                "{dtype:?} {tile:?} static shared memory"
            );
            assert_eq!(
                preferred_carveout, -1,
                "{dtype:?} {tile:?} must retain the CUDA default shared-memory carveout"
            );
            assert!(
                max_dynamic_shared_bytes
                    >= i32::try_from(dynamic_shared_bytes)
                        .expect("SM120 half dynamic shared memory fits i32"),
                "{dtype:?} {tile:?} max dynamic shared-memory opt-in {max_dynamic_shared_bytes} is below launch requirement {dynamic_shared_bytes}"
            );
            assert!(occupancy >= 2, "{dtype:?} {tile:?} residency");
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits selector performance data"]
fn fixed_thin_selector_smoke() {
    let shapes = [
        InferenceShape {
            m: 65,
            k: 768,
            n: 1536,
        },
        InferenceShape {
            m: 96,
            k: 768,
            n: 1536,
        },
        InferenceShape {
            m: 128,
            k: 768,
            n: 1536,
        },
        InferenceShape {
            m: 128,
            k: 1536,
            n: 1536,
        },
        InferenceShape {
            m: 128,
            k: 2560,
            n: 1536,
        },
        InferenceShape {
            m: 128,
            k: 768,
            n: 1928,
        },
        InferenceShape {
            m: 128,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 160,
            k: 1536,
            n: 1536,
        },
        InferenceShape {
            m: 129,
            k: 1537,
            n: 1535,
        },
    ];
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
            let operands = InferenceFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            for tile in [
                InferenceTile::Tc16,
                InferenceTile::Tc64,
                InferenceTile::Tc128,
            ] {
                let elapsed_us = average_us(&ctx, || {
                    inference_forward_with_tile(&ctx, operands, shape, tile)
                        .unwrap_or_else(|error| panic!("forced {tile:?}: {error}"));
                });
                println!(
                    "dtype={dtype:?} m={} k={} n={} tile={tile:?} elapsed_us={elapsed_us:.3}",
                    shape.m, shape.k, shape.n,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits dense selector data"]
fn fixed_thin_selector_grid_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for m in [65usize, 80, 96, 112, 128, 129, 160] {
            for n in [512usize, 1024, 1536, 1928, 2304] {
                let shape = InferenceShape { m, k: 768, n };
                let a =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
                let b =
                    DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
                let c =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
                let operands = InferenceFwdOperands {
                    c: typed(&c, dtype),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr: None,
                };
                let thin_us = average_us(&ctx, || {
                    inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc16)
                        .expect("forced Tc16");
                });
                let square_us = average_us(&ctx, || {
                    inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc64)
                        .expect("forced Tc64");
                });
                println!(
                    "dtype={dtype:?} m={m} k=768 n={n} tc16_us={thin_us:.3} tc64_us={square_us:.3} tc16_over_tc64={:.5}",
                    thin_us / square_us,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits selector boundary data"]
fn fixed_thin_selector_large_m_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for m in [192usize, 256, 320, 384, 512, 768] {
            for n in [512usize, 1024, 1536] {
                let shape = InferenceShape { m, k: 768, n };
                let a =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
                let b =
                    DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
                let c =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
                let operands = InferenceFwdOperands {
                    c: typed(&c, dtype),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr: None,
                };
                let thin_us = average_us(&ctx, || {
                    inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc16)
                        .expect("forced Tc16");
                });
                let square_us = average_us(&ctx, || {
                    inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc64)
                        .expect("forced Tc64");
                });
                println!(
                    "dtype={dtype:?} m={m} k=768 n={n} tc16_us={thin_us:.3} tc64_us={square_us:.3} tc16_over_tc64={:.5}",
                    thin_us / square_us,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an SM80+ CUDA device and emits selector K-axis data"]
fn fixed_thin_selector_k_axis_smoke() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let mn_pairs = [
        (256usize, 512usize),
        (320, 512),
        (128, 1024),
        (160, 1024),
        (96, 1536),
        (112, 1536),
    ];
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, n) in mn_pairs {
            for k in [64usize, 128, 256, 384, 768, 1536, 2560] {
                let shape = InferenceShape { m, k, n };
                let a =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
                let b =
                    DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
                let c =
                    DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
                let operands = InferenceFwdOperands {
                    c: typed(&c, dtype),
                    x: typed(&a, dtype),
                    w: typed(&b, dtype),
                    bias_ptr: None,
                };
                let thin_us = average_us(&ctx, || {
                    inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc16)
                        .expect("forced Tc16");
                });
                let square_us = average_us(&ctx, || {
                    inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc64)
                        .expect("forced Tc64");
                });
                println!(
                    "dtype={dtype:?} m={m} k={k} n={n} tc16_us={thin_us:.3} tc64_us={square_us:.3} tc16_over_tc64={:.5}",
                    thin_us / square_us,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a quiet SM89 CUDA device and emits paired selector evidence"]
fn fixed_thin_selector_sm89_paired_boundary() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9), "SM89 required");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    let shapes = [
        InferenceShape {
            m: 96,
            k: 64,
            n: 1536,
        },
        InferenceShape {
            m: 96,
            k: 768,
            n: 1536,
        },
        InferenceShape {
            m: 96,
            k: 2560,
            n: 1536,
        },
        InferenceShape {
            m: 112,
            k: 768,
            n: 1536,
        },
        InferenceShape {
            m: 112,
            k: 1536,
            n: 1536,
        },
        InferenceShape {
            m: 128,
            k: 768,
            n: 1536,
        },
        InferenceShape {
            m: 129,
            k: 768,
            n: 1024,
        },
        InferenceShape {
            m: 160,
            k: 768,
            n: 1024,
        },
        InferenceShape {
            m: 160,
            k: 1536,
            n: 1024,
        },
        InferenceShape {
            m: 160,
            k: 2560,
            n: 1024,
        },
        InferenceShape {
            m: 320,
            k: 768,
            n: 512,
        },
        InferenceShape {
            m: 336,
            k: 768,
            n: 512,
        },
        InferenceShape {
            m: 512,
            k: 768,
            n: 512,
        },
        InferenceShape {
            m: 576,
            k: 768,
            n: 512,
        },
        InferenceShape {
            m: 256,
            k: 768,
            n: 1024,
        },
        InferenceShape {
            m: 288,
            k: 768,
            n: 1024,
        },
        InferenceShape {
            m: 176,
            k: 768,
            n: 1536,
        },
        InferenceShape {
            m: 192,
            k: 768,
            n: 1536,
        },
        InferenceShape {
            m: 96,
            k: 768,
            n: 1928,
        },
        InferenceShape {
            m: 112,
            k: 768,
            n: 1928,
        },
        InferenceShape {
            m: 128,
            k: 768,
            n: 1928,
        },
        InferenceShape {
            m: 112,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 128,
            k: 768,
            n: 2304,
        },
    ];

    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
            let operands = InferenceFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            for _ in 0..64 {
                inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc16)
                    .expect("warm Tc16");
                inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc64)
                    .expect("warm Tc64");
            }
            ctx.stream.synchronize().expect("selector warmup sync");
            let thin_iterations =
                fixed_tile_window_iterations(&ctx, operands, shape, InferenceTile::Tc16);
            let square_iterations =
                fixed_tile_window_iterations(&ctx, operands, shape, InferenceTile::Tc64);
            let mut ratios = Vec::with_capacity(101);
            for window in 0..101 {
                let (thin_us, square_us) = if window % 2 == 0 {
                    (
                        fixed_tile_window_us(
                            &ctx,
                            operands,
                            shape,
                            InferenceTile::Tc16,
                            thin_iterations,
                        ),
                        fixed_tile_window_us(
                            &ctx,
                            operands,
                            shape,
                            InferenceTile::Tc64,
                            square_iterations,
                        ),
                    )
                } else {
                    let square_us = fixed_tile_window_us(
                        &ctx,
                        operands,
                        shape,
                        InferenceTile::Tc64,
                        square_iterations,
                    );
                    let thin_us = fixed_tile_window_us(
                        &ctx,
                        operands,
                        shape,
                        InferenceTile::Tc16,
                        thin_iterations,
                    );
                    (thin_us, square_us)
                };
                ratios.push(thin_us / square_us);
            }
            ratios.sort_by(f64::total_cmp);
            println!(
                concat!(
                    "{{\"schema\":\"MambaBiFixedThinSelectorPairedV1\",",
                    "\"dtype\":\"{:?}\",\"m\":{},\"k\":{},\"n\":{},",
                    "\"windows\":101,\"thin_iterations\":{},\"square_iterations\":{},",
                    "\"ratio_p05\":{:.9},\"ratio_p50\":{:.9},\"ratio_p95\":{:.9}}}"
                ),
                dtype,
                shape.m,
                shape.k,
                shape.n,
                thin_iterations,
                square_iterations,
                percentile(&ratios, 0.05),
                percentile(&ratios, 0.50),
                percentile(&ratios, 0.95),
            );
        }
    }
}

#[test]
#[ignore = "requires a quiet SM89 CUDA device and emits Tc16 occupancy evidence"]
fn fixed_tc16_occupancy_cliff_paired_census() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert_eq!(
        ctx.stream
            .context()
            .compute_capability()
            .expect("CUDA compute capability"),
        (8, 9),
        "Tc16 occupancy inventory is qualified on SM89",
    );
    assert_eq!(device.multiprocessor_count(), 142);
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    let shapes = [
        InferenceShape {
            m: 64,
            k: 768,
            n: 3392,
        },
        InferenceShape {
            m: 64,
            k: 768,
            n: 3424,
        },
        InferenceShape {
            m: 64,
            k: 768,
            n: 4544,
        },
        InferenceShape {
            m: 64,
            k: 1536,
            n: 3424,
        },
        InferenceShape {
            m: 1,
            k: 768,
            n: 32768,
        },
        InferenceShape {
            m: 64,
            k: 768,
            n: 4096,
        },
    ];
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for shape in shapes {
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, dtype).expect("A allocation");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, dtype).expect("B allocation");
            let c = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, dtype).expect("C allocation");
            let operands = InferenceFwdOperands {
                c: typed(&c, dtype),
                x: typed(&a, dtype),
                w: typed(&b, dtype),
                bias_ptr: None,
            };
            for _ in 0..10 {
                inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc16)
                    .expect("Tc16 warmup");
                inference_forward_with_tile(&ctx, operands, shape, InferenceTile::Tc64)
                    .expect("Tc64 warmup");
            }
            ctx.stream.synchronize().expect("occupancy warmup sync");
            let iterations =
                fixed_tile_window_iterations(&ctx, operands, shape, InferenceTile::Tc16).max(
                    fixed_tile_window_iterations(&ctx, operands, shape, InferenceTile::Tc64),
                );
            let mut ratios = Vec::with_capacity(101);
            for round in 0..101 {
                let (tc16_us, tc64_us) = if round & 1 == 0 {
                    let thin = fixed_tile_window_us(
                        &ctx,
                        operands,
                        shape,
                        InferenceTile::Tc16,
                        iterations,
                    );
                    let square = fixed_tile_window_us(
                        &ctx,
                        operands,
                        shape,
                        InferenceTile::Tc64,
                        iterations,
                    );
                    (thin, square)
                } else {
                    let square = fixed_tile_window_us(
                        &ctx,
                        operands,
                        shape,
                        InferenceTile::Tc64,
                        iterations,
                    );
                    let thin = fixed_tile_window_us(
                        &ctx,
                        operands,
                        shape,
                        InferenceTile::Tc16,
                        iterations,
                    );
                    (thin, square)
                };
                ratios.push(tc16_us / tc64_us);
            }
            ratios.sort_by(f64::total_cmp);
            println!(
                "dtype={dtype:?} m={} k={} n={} iterations={} tc16_over_tc64_p05={:.6} p50={:.6} p95={:.6}",
                shape.m,
                shape.k,
                shape.n,
                iterations,
                percentile(&ratios, 0.05),
                percentile(&ratios, 0.50),
                percentile(&ratios, 0.95),
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FixedSm120Tf32BdComparison {
    id: &'static str,
    cell: &'static str,
    shape: InferenceShape,
    dtype: WeightDtype,
    incumbent: InferenceTile,
    candidate: InferenceTile,
    incumbent_gap_close_ratio: f64,
}

const FIXED_SM120_TF32_BD_PAIRED_COMPARISONS: [FixedSm120Tf32BdComparison; 3] = [
    FixedSm120Tf32BdComparison {
        id: "b_m128_vs_sm120_m64",
        cell: "B",
        shape: InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        dtype: WeightDtype::F32,
        incumbent: InferenceTile::Tf32Sm120M64S2,
        candidate: InferenceTile::Tf32Sm120M128S2,
        incumbent_gap_close_ratio: 0.936040362,
    },
    FixedSm120Tf32BdComparison {
        id: "d_m128_vs_sm120_m64",
        cell: "D",
        shape: InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        dtype: WeightDtype::F32,
        incumbent: InferenceTile::Tf32Sm120M64S2,
        candidate: InferenceTile::Tf32Sm120M128S2,
        incumbent_gap_close_ratio: 0.974839395,
    },
    FixedSm120Tf32BdComparison {
        id: "d_portable_m64_vs_sm120_m64",
        cell: "D",
        shape: InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        dtype: WeightDtype::F32,
        incumbent: InferenceTile::Tf32Sm120M64S2,
        candidate: InferenceTile::Tf32M64S2,
        incumbent_gap_close_ratio: 0.0,
    },
];

const FIXED_SM120_TF32_BD_WARMUPS: usize = 128;
const FIXED_SM120_TF32_BD_PILOT_ITERATIONS: usize = 16;
const FIXED_SM120_TF32_BD_TARGET_WINDOW_MS: f64 = 5.0;
const FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER: usize = 101;
const FIXED_SM120_TF32_BD_ROUTE_ORDERS: [&str; 2] =
    ["candidate_then_incumbent", "incumbent_then_candidate"];
const FIXED_SM120_TF32_BD_BIASES: [&str; 2] = ["none", "synthesized"];
const FIXED_SM120_TF32_BD_VENDOR_ORDERS: [&str; 2] =
    ["candidate_then_vendor", "vendor_then_candidate"];
const FIXED_SM120_TF32_BD_SCHEMA: &str = "MambaBiFixedTf32BdForcedPairedV1";
const FIXED_SM120_TF32_BD_SUITE: &str = "fixed_sm120_tf32_bd_paired_forced_routes";
const FIXED_SM120_TF32_BD_JSONL_ENV: &str = "GEMM_BI_TF32_BD_JSONL";
const FIXED_SM120_TF32_BD_EXPECTED_PORTABLE_REJECTION_GATE: &str = "resource_registers";
const FIXED_SM120_TF32_BD_A_SEED: u64 = 0xa170_7f32_0000_0001;
const FIXED_SM120_TF32_BD_B_SEED: u64 = 0xb170_7f32_0000_0002;
const FIXED_SM120_TF32_BD_BIAS_SEED: u64 = 0xb1a5_7f32_0000_0003;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FixedSm120Tf32BdRouteSpec {
    tile_label: &'static str,
    symbol: &'static str,
    block_m: usize,
    block_n: usize,
    block_threads: u32,
    dynamic_shared_bytes: usize,
    minimum_occupancy: u32,
}

#[derive(Debug, Clone, Copy)]
struct FixedSm120Tf32BdResourceSnapshot {
    spec: FixedSm120Tf32BdRouteSpec,
    grid: usize,
    registers: i32,
    local_bytes: i32,
    static_shared_bytes: i32,
    occupancy_blocks_per_sm: u32,
    preferred_shared_memory_carveout: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FixedSm120Tf32BdResourceDisposition {
    Pass,
    ExpectedPortableRegisterRejection,
    UnexpectedResourceDefect {
        failed_gate: &'static str,
        reason: String,
    },
}

#[derive(Debug, Default)]
struct FixedSm120Tf32BdRecordCounts {
    preflight: Cell<usize>,
    bit_gate: Cell<usize>,
    window: Cell<usize>,
    summary: Cell<usize>,
    rejection: Cell<usize>,
}

impl FixedSm120Tf32BdRecordCounts {
    fn increment(&self, record_type: &str) {
        let counter = match record_type {
            "preflight" => &self.preflight,
            "bit_gate" => &self.bit_gate,
            "window" => &self.window,
            "summary" => &self.summary,
            "rejection" => &self.rejection,
            _ => panic!("unknown TF32 B/D record type {record_type}"),
        };
        counter.set(counter.get() + 1);
    }

    fn has_no_measurement_records(&self) -> bool {
        self.bit_gate.get() == 0 && self.window.get() == 0 && self.summary.get() == 0
    }
}

struct FixedSm120Tf32BdJsonlSink {
    writer: RefCell<BufWriter<File>>,
}

impl FixedSm120Tf32BdJsonlSink {
    fn from_env() -> Result<Self, String> {
        let path = std::env::var_os(FIXED_SM120_TF32_BD_JSONL_ENV)
            .ok_or_else(|| format!("{FIXED_SM120_TF32_BD_JSONL_ENV} must name the JSONL sink"))?;
        if path.is_empty() {
            return Err(format!("{FIXED_SM120_TF32_BD_JSONL_ENV} must not be empty"));
        }
        let file = File::create(&path)
            .map_err(|error| format!("create TF32 B/D JSONL sink {path:?}: {error}"))?;
        eprintln!("TF32 B/D JSONL sink: {path:?}");
        Ok(Self {
            writer: RefCell::new(BufWriter::new(file)),
        })
    }

    fn emit(&self, record: &str) {
        let mut writer = self.writer.borrow_mut();
        writeln!(writer, "{record}").expect("write TF32 B/D JSONL record");
        writer.flush().expect("flush TF32 B/D JSONL record");
        println!("{record}");
    }
}

struct FixedSm120Tf32BdComparisonEmitter<'a> {
    sink: &'a FixedSm120Tf32BdJsonlSink,
    counts: FixedSm120Tf32BdRecordCounts,
}

impl<'a> FixedSm120Tf32BdComparisonEmitter<'a> {
    fn new(sink: &'a FixedSm120Tf32BdJsonlSink) -> Self {
        Self {
            sink,
            counts: FixedSm120Tf32BdRecordCounts::default(),
        }
    }

    fn emit(&self, record_type: &str, record: String) {
        self.counts.increment(record_type);
        self.sink.emit(&record);
    }
}

#[derive(Debug, Clone)]
struct FixedSm120Tf32BdSummary {
    comparison_kind: &'static str,
    comparator_order: &'static str,
    candidate_p05_us: f64,
    candidate_p50_us: f64,
    candidate_p95_us: f64,
    comparator_p05_us: f64,
    comparator_p50_us: f64,
    comparator_p95_us: f64,
    ratio_p05: f64,
    ratio_p50: f64,
    ratio_p95: f64,
}

#[derive(Clone, Copy)]
struct FixedSm120Tf32BdRecordContext<'a> {
    comparison: FixedSm120Tf32BdComparison,
    bias: &'a str,
    device_cc: (u32, u32),
    sm_count: u32,
    emitter: &'a FixedSm120Tf32BdComparisonEmitter<'a>,
}

fn fixed_sm120_tf32_bd_route_spec(tile: InferenceTile) -> FixedSm120Tf32BdRouteSpec {
    match tile {
        InferenceTile::Tf32Sm120M64S2 => FixedSm120Tf32BdRouteSpec {
            tile_label: "Tf32Sm120M64S2",
            symbol: "nn_sm120_tma_tf32_m64n64_bk32_s2",
            block_m: 64,
            block_n: 64,
            block_threads: 128,
            dynamic_shared_bytes: 32_896,
            minimum_occupancy: 3,
        },
        InferenceTile::Tf32Sm120M128S2 => FixedSm120Tf32BdRouteSpec {
            tile_label: "Tf32Sm120M128S2",
            symbol: "nn_sm120_tma_tf32_m128n64_bk32_s2",
            block_m: 128,
            block_n: 64,
            block_threads: 128,
            dynamic_shared_bytes: 49_280,
            minimum_occupancy: 2,
        },
        InferenceTile::Tf32M64S2 => FixedSm120Tf32BdRouteSpec {
            tile_label: "Tf32M64S2",
            symbol: "nn_tf32_m64n64_bk32_s2",
            block_m: 64,
            block_n: 64,
            block_threads: 128,
            dynamic_shared_bytes: 32_768,
            minimum_occupancy: 3,
        },
        _ => panic!("{tile:?} is outside the frozen TF32 B/D paired inventory"),
    }
}

fn fixed_sm120_tf32_bd_resource_snapshot(
    ctx: &GpuCtx,
    shape: InferenceShape,
    tile: InferenceTile,
) -> Result<FixedSm120Tf32BdResourceSnapshot, String> {
    let spec = fixed_sm120_tf32_bd_route_spec(tile);
    let function = match tile {
        InferenceTile::Tf32Sm120M64S2 => {
            &ctx.kernels
                .gemm_bi_nn_tf32_sm120
                .as_ref()
                .ok_or("Fixed SM120 TF32 function holder is absent")?
                .m64n64_s2
        }
        InferenceTile::Tf32Sm120M128S2 => {
            &ctx.kernels
                .gemm_bi_nn_tf32_sm120
                .as_ref()
                .ok_or("Fixed SM120 TF32 function holder is absent")?
                .m128n64_s2
        }
        InferenceTile::Tf32M64S2 => &ctx.kernels.gemm_bi_nn_tf32.m64n64_s2,
        _ => {
            return Err(format!(
                "{tile:?} is outside the TF32 B/D resource inventory"
            ));
        }
    };
    let registers = function
        .num_regs()
        .map_err(|error| format!("read {} registers: {error:?}", spec.symbol))?;
    let local_bytes = function
        .local_size_bytes()
        .map_err(|error| format!("read {} local bytes: {error:?}", spec.symbol))?;
    let static_shared_bytes = function
        .shared_size_bytes()
        .map_err(|error| format!("read {} static shared bytes: {error:?}", spec.symbol))?;
    let occupancy_blocks_per_sm = function
        .occupancy_max_active_blocks_per_multiprocessor(
            spec.block_threads,
            spec.dynamic_shared_bytes,
            None,
        )
        .map_err(|error| format!("read {} occupancy: {error:?}", spec.symbol))?;
    let preferred_shared_memory_carveout = function
        .get_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        )
        .map_err(|error| format!("read {} preferred carveout: {error:?}", spec.symbol))?;
    Ok(FixedSm120Tf32BdResourceSnapshot {
        spec,
        grid: shape.m.div_ceil(spec.block_m) * shape.n.div_ceil(spec.block_n),
        registers,
        local_bytes,
        static_shared_bytes,
        occupancy_blocks_per_sm,
        preferred_shared_memory_carveout,
    })
}

fn fixed_sm120_tf32_bd_resource_disposition(
    comparison: FixedSm120Tf32BdComparison,
    arm: &str,
    tile: InferenceTile,
    snapshot: FixedSm120Tf32BdResourceSnapshot,
) -> FixedSm120Tf32BdResourceDisposition {
    let expected = fixed_sm120_tf32_bd_route_spec(tile);
    if snapshot.spec != expected
        || snapshot.grid
            != comparison.shape.m.div_ceil(expected.block_m)
                * comparison.shape.n.div_ceil(expected.block_n)
    {
        return FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect {
            failed_gate: "resource_mapping",
            reason: format!(
                "{arm} route metadata is incoherent: expected {expected:?}, observed {snapshot:?}"
            ),
        };
    }
    if snapshot.registers <= 0 {
        return FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect {
            failed_gate: "resource_registers",
            reason: format!(
                "{arm} {} reported invalid register count {}",
                snapshot.spec.symbol, snapshot.registers
            ),
        };
    }
    if snapshot.local_bytes != 0 {
        return FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect {
            failed_gate: "resource_local_bytes",
            reason: format!(
                "{arm} {} uses {} local bytes, required 0",
                snapshot.spec.symbol, snapshot.local_bytes
            ),
        };
    }
    if snapshot.occupancy_blocks_per_sm < expected.minimum_occupancy {
        return FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect {
            failed_gate: "resource_occupancy",
            reason: format!(
                "{arm} {} occupancy {} is below required {}",
                snapshot.spec.symbol, snapshot.occupancy_blocks_per_sm, expected.minimum_occupancy
            ),
        };
    }
    if snapshot.registers > 128 {
        if comparison.id == "d_portable_m64_vs_sm120_m64"
            && arm == "candidate"
            && tile == InferenceTile::Tf32M64S2
            && snapshot.spec.symbol == "nn_tf32_m64n64_bk32_s2"
            && snapshot.spec.dynamic_shared_bytes == 32_768
        {
            return FixedSm120Tf32BdResourceDisposition::ExpectedPortableRegisterRejection;
        }
        return FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect {
            failed_gate: "resource_registers",
            reason: format!(
                "{arm} {} uses {} registers, limit is 128",
                snapshot.spec.symbol, snapshot.registers
            ),
        };
    }
    FixedSm120Tf32BdResourceDisposition::Pass
}

fn fixed_sm120_tf32_bd_digest(bits: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for word in bits {
        hasher.update(word.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn fixed_sm120_tf32_bd_json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => escaped.push(character),
        }
    }
    escaped
}

fn fixed_sm120_tf32_bd_common_json(
    comparison: FixedSm120Tf32BdComparison,
    bias: &str,
    device_cc: (u32, u32),
    sm_count: u32,
    incumbent_iterations: usize,
    candidate_iterations: usize,
) -> String {
    let incumbent = fixed_sm120_tf32_bd_route_spec(comparison.incumbent);
    let candidate = fixed_sm120_tf32_bd_route_spec(comparison.candidate);
    format!(
        concat!(
            "\"schema\":\"{}\",\"suite\":\"{}\",",
            "\"comparison_id\":\"{}\",\"cell\":\"{}\",",
            "\"shape\":{{\"m\":{},\"k\":{},\"n\":{}}},",
            "\"device\":{{\"cc\":\"{}.{}\",\"sm_count\":{}}},",
            "\"dtype\":\"f32\",\"f32_policy\":\"allow_deterministic_tf32\",",
            "\"bias\":\"{}\",",
            "\"seed\":{{\"a\":\"0x{:016x}\",\"b\":\"0x{:016x}\",\"bias\":\"0x{:016x}\"}},",
            "\"input_lengths\":{{\"a\":{},\"b\":{},\"bias\":{}}},\"output_len\":{},",
            "\"warmups\":{},\"pilot_iterations\":{},\"target_window_ms\":{:.1},",
            "\"windows_per_order\":{},",
            "\"incumbent\":{{\"tile\":\"{}\",\"symbol\":\"{}\",\"iterations\":{}}},",
            "\"candidate\":{{\"tile\":\"{}\",\"symbol\":\"{}\",\"iterations\":{}}}"
        ),
        FIXED_SM120_TF32_BD_SCHEMA,
        FIXED_SM120_TF32_BD_SUITE,
        comparison.id,
        comparison.cell,
        comparison.shape.m,
        comparison.shape.k,
        comparison.shape.n,
        device_cc.0,
        device_cc.1,
        sm_count,
        bias,
        FIXED_SM120_TF32_BD_A_SEED,
        FIXED_SM120_TF32_BD_B_SEED,
        FIXED_SM120_TF32_BD_BIAS_SEED,
        comparison.shape.m * comparison.shape.k,
        comparison.shape.k * comparison.shape.n,
        comparison.shape.n,
        comparison.shape.m * comparison.shape.n,
        FIXED_SM120_TF32_BD_WARMUPS,
        FIXED_SM120_TF32_BD_PILOT_ITERATIONS,
        FIXED_SM120_TF32_BD_TARGET_WINDOW_MS,
        FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER,
        incumbent.tile_label,
        incumbent.symbol,
        incumbent_iterations,
        candidate.tile_label,
        candidate.symbol,
        candidate_iterations,
    )
}

fn fixed_sm120_tf32_bd_reject(
    comparison: FixedSm120Tf32BdComparison,
    bias: &str,
    device_cc: (u32, u32),
    sm_count: u32,
    failed_gate: &str,
    reason: impl AsRef<str>,
    emitter: &FixedSm120Tf32BdComparisonEmitter<'_>,
) -> ! {
    emitter.emit(
        "rejection",
        format!(
            "{{{},\"record_type\":\"rejection\",\"failed_gate\":\"{}\",\"reason\":\"{}\",\"promotion_eligible\":false}}",
            fixed_sm120_tf32_bd_common_json(comparison, bias, device_cc, sm_count, 0, 0),
            fixed_sm120_tf32_bd_json_escape(failed_gate),
            fixed_sm120_tf32_bd_json_escape(reason.as_ref()),
        ),
    );
    panic!(
        "TF32 B/D comparison {} rejected at {}: {}",
        comparison.id,
        failed_gate,
        reason.as_ref()
    );
}

fn fixed_sm120_tf32_bd_emit_bit_gate(
    record: FixedSm120Tf32BdRecordContext<'_>,
    arm: &str,
    phase: &str,
    replay_index: Option<usize>,
    output_sha256: &str,
    reference_sha256: &str,
) {
    let replay_index = replay_index
        .map(|index| index.to_string())
        .unwrap_or_else(|| "null".to_owned());
    record.emitter.emit(
        "bit_gate",
        format!(
            "{{{},\"record_type\":\"bit_gate\",\"arm\":\"{}\",\"phase\":\"{}\",\"replay_index\":{},\"output_sha256\":\"{}\",\"reference_sha256\":\"{}\",\"exact_bits_equal\":true,\"passed\":true}}",
            fixed_sm120_tf32_bd_common_json(
                record.comparison,
                record.bias,
                record.device_cc,
                record.sm_count,
                0,
                0,
            ),
            arm,
            phase,
            replay_index,
            output_sha256,
            reference_sha256,
        ),
    );
}

fn fixed_sm120_tf32_bd_forced_window_us(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    tile: InferenceTile,
    iterations: usize,
) -> f64 {
    configure_fixed_auto_vendor_custom(ctx, F32TriadPolicy::AllowDeterministicTf32);
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record forced Fixed window start");
    for _ in 0..iterations {
        inference_forward_with_tile(ctx, operands, shape, tile)
            .unwrap_or_else(|error| panic!("forced Fixed {tile:?}: {error}"));
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record forced Fixed window end");
    f64::from(start.elapsed_ms(&end).expect("measure forced Fixed window")) * 1000.0
        / iterations as f64
}

fn fixed_sm120_tf32_bd_forced_iterations(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    tile: InferenceTile,
) -> usize {
    let pilot_us = fixed_sm120_tf32_bd_forced_window_us(
        ctx,
        operands,
        shape,
        tile,
        FIXED_SM120_TF32_BD_PILOT_ITERATIONS,
    );
    fixed_auto_vendor_iterations(pilot_us)
}

fn fixed_sm120_tf32_bd_environment_preflight(label: &str) -> Result<(), String> {
    let applications = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|error| format!("run compute-application preflight: {error}"))?;
    if !applications.status.success() {
        return Err(format!(
            "compute-application preflight exited with {}",
            applications.status
        ));
    }
    let own_pid = std::process::id();
    let competing_pids = String::from_utf8_lossy(&applications.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .filter(|&pid| pid != own_pid)
        .collect::<Vec<_>>();
    if !competing_pids.is_empty() {
        return Err(format!(
            "competing compute applications: {competing_pids:?}"
        ));
    }

    let mut last_snapshot = None;
    for _ in 0..50 {
        let snapshot = Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,clocks.sm,temperature.gpu,pstate",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|error| format!("run GPU telemetry preflight: {error}"))?;
        if !snapshot.status.success() {
            return Err(format!(
                "GPU telemetry preflight exited with {}",
                snapshot.status
            ));
        }
        let line = String::from_utf8_lossy(&snapshot.stdout)
            .lines()
            .next()
            .ok_or("GPU telemetry preflight returned no rows")?
            .to_owned();
        let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 6 {
            return Err(format!(
                "GPU telemetry preflight returned malformed row {line:?}"
            ));
        }
        let gpu_util = fields[0]
            .parse::<u32>()
            .map_err(|error| format!("parse GPU utilization {:?}: {error}", fields[0]))?;
        let memory_util = fields[1]
            .parse::<u32>()
            .map_err(|error| format!("parse memory utilization {:?}: {error}", fields[1]))?;
        let used_mib = fields[2]
            .parse::<u32>()
            .map_err(|error| format!("parse used memory {:?}: {error}", fields[2]))?;
        let sm_clock_mhz = fields[3]
            .parse::<u32>()
            .map_err(|error| format!("parse SM clock {:?}: {error}", fields[3]))?;
        let temperature_c = fields[4]
            .parse::<u32>()
            .map_err(|error| format!("parse GPU temperature {:?}: {error}", fields[4]))?;
        let pstate = fields[5];
        last_snapshot = Some(format!(
            "gpu_util={gpu_util}% memory_util={memory_util}% used={used_mib}MiB sm_clock={sm_clock_mhz}MHz temperature={temperature_c}C pstate={pstate}"
        ));
        if gpu_util <= 1 && memory_util <= 1 && sm_clock_mhz > 0 && temperature_c > 0 {
            eprintln!(
                "TF32 B/D route-set preflight {label}: {} (the process's own allocations are excluded from the launch-time <=128 MiB gate)",
                last_snapshot.as_deref().unwrap_or("telemetry unavailable")
            );
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    Err(format!(
        "GPU did not return to <=1% compute and memory utilization; last snapshot: {}",
        last_snapshot.as_deref().unwrap_or("none")
    ))
}

fn fixed_sm120_tf32_bd_summary(
    comparison_kind: &'static str,
    comparator_order: &'static str,
    candidate_us: &[f64],
    comparator_us: &[f64],
    ratios: &[f64],
) -> FixedSm120Tf32BdSummary {
    let mut candidate_sorted = candidate_us.to_vec();
    let mut comparator_sorted = comparator_us.to_vec();
    let mut ratio_sorted = ratios.to_vec();
    candidate_sorted.sort_by(f64::total_cmp);
    comparator_sorted.sort_by(f64::total_cmp);
    ratio_sorted.sort_by(f64::total_cmp);
    FixedSm120Tf32BdSummary {
        comparison_kind,
        comparator_order,
        candidate_p05_us: percentile(&candidate_sorted, 0.05),
        candidate_p50_us: percentile(&candidate_sorted, 0.50),
        candidate_p95_us: percentile(&candidate_sorted, 0.95),
        comparator_p05_us: percentile(&comparator_sorted, 0.05),
        comparator_p50_us: percentile(&comparator_sorted, 0.50),
        comparator_p95_us: percentile(&comparator_sorted, 0.95),
        ratio_p05: percentile(&ratio_sorted, 0.05),
        ratio_p50: percentile(&ratio_sorted, 0.50),
        ratio_p95: percentile(&ratio_sorted, 0.95),
    }
}

#[test]
fn fixed_sm120_tf32_bd_paired_inventory_is_frozen() {
    assert_eq!(FIXED_SM120_TF32_BD_PAIRED_COMPARISONS.len(), 3);
    assert_eq!(
        FIXED_SM120_TF32_BD_PAIRED_COMPARISONS,
        [
            FixedSm120Tf32BdComparison {
                id: "b_m128_vs_sm120_m64",
                cell: "B",
                shape: InferenceShape {
                    m: 4621,
                    k: 768,
                    n: 2304,
                },
                dtype: WeightDtype::F32,
                incumbent: InferenceTile::Tf32Sm120M64S2,
                candidate: InferenceTile::Tf32Sm120M128S2,
                incumbent_gap_close_ratio: 0.936040362,
            },
            FixedSm120Tf32BdComparison {
                id: "d_m128_vs_sm120_m64",
                cell: "D",
                shape: InferenceShape {
                    m: 2048,
                    k: 768,
                    n: 2304,
                },
                dtype: WeightDtype::F32,
                incumbent: InferenceTile::Tf32Sm120M64S2,
                candidate: InferenceTile::Tf32Sm120M128S2,
                incumbent_gap_close_ratio: 0.974839395,
            },
            FixedSm120Tf32BdComparison {
                id: "d_portable_m64_vs_sm120_m64",
                cell: "D",
                shape: InferenceShape {
                    m: 2048,
                    k: 768,
                    n: 2304,
                },
                dtype: WeightDtype::F32,
                incumbent: InferenceTile::Tf32Sm120M64S2,
                candidate: InferenceTile::Tf32M64S2,
                incumbent_gap_close_ratio: 0.0,
            },
        ]
    );
    assert_eq!(FIXED_SM120_TF32_BD_WARMUPS, 128);
    assert_eq!(FIXED_SM120_TF32_BD_PILOT_ITERATIONS, 16);
    assert_eq!(FIXED_SM120_TF32_BD_TARGET_WINDOW_MS, 5.0);
    assert_eq!(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER, 101);
    assert_eq!(
        FIXED_SM120_TF32_BD_ROUTE_ORDERS,
        ["candidate_then_incumbent", "incumbent_then_candidate"]
    );
    assert_eq!(FIXED_SM120_TF32_BD_BIASES, ["none", "synthesized"]);
    assert_eq!(
        FIXED_SM120_TF32_BD_VENDOR_ORDERS,
        ["candidate_then_vendor", "vendor_then_candidate"]
    );
    let mut ids = FIXED_SM120_TF32_BD_PAIRED_COMPARISONS
        .iter()
        .map(|comparison| comparison.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 3, "comparison ids must be unique");
}

#[test]
fn fixed_sm120_tf32_bd_expected_portable_rejection_is_terminal() {
    let comparison = FIXED_SM120_TF32_BD_PAIRED_COMPARISONS[2];
    let sealed_snapshot = FixedSm120Tf32BdResourceSnapshot {
        spec: fixed_sm120_tf32_bd_route_spec(InferenceTile::Tf32M64S2),
        grid: 1152,
        registers: 167,
        local_bytes: 0,
        static_shared_bytes: 0,
        occupancy_blocks_per_sm: 3,
        preferred_shared_memory_carveout: -1,
    };
    assert_eq!(
        fixed_sm120_tf32_bd_resource_disposition(
            comparison,
            "candidate",
            InferenceTile::Tf32M64S2,
            sealed_snapshot,
        ),
        FixedSm120Tf32BdResourceDisposition::ExpectedPortableRegisterRejection
    );
    assert_eq!(
        fixed_sm120_tf32_bd_resource_disposition(
            comparison,
            "candidate",
            InferenceTile::Tf32M64S2,
            FixedSm120Tf32BdResourceSnapshot {
                registers: 129,
                ..sealed_snapshot
            },
        ),
        FixedSm120Tf32BdResourceDisposition::ExpectedPortableRegisterRejection,
        "the durable rejection law is registers > 128, not exactly 167"
    );
    assert_eq!(
        fixed_sm120_tf32_bd_resource_disposition(
            comparison,
            "candidate",
            InferenceTile::Tf32M64S2,
            FixedSm120Tf32BdResourceSnapshot {
                registers: 128,
                ..sealed_snapshot
            },
        ),
        FixedSm120Tf32BdResourceDisposition::Pass
    );
    assert!(matches!(
        fixed_sm120_tf32_bd_resource_disposition(
            comparison,
            "candidate",
            InferenceTile::Tf32M64S2,
            FixedSm120Tf32BdResourceSnapshot {
                local_bytes: 4,
                ..sealed_snapshot
            },
        ),
        FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect { .. }
    ));
    let counts = FixedSm120Tf32BdRecordCounts::default();
    assert!(counts.has_no_measurement_records());
    assert_eq!(
        FIXED_SM120_TF32_BD_EXPECTED_PORTABLE_REJECTION_GATE,
        "resource_registers"
    );
}

#[test]
#[ignore = "requires a quiet 170-SM CC12.0 CUDA device and emits raw paired TF32 B/D evidence"]
fn fixed_sm120_tf32_bd_paired_forced_routes() {
    let sink = FixedSm120Tf32BdJsonlSink::from_env().unwrap_or_else(|error| panic!("{error}"));
    let device = GpuDevice::new(0).expect("CUDA device");
    let device_cc = device.compute_capability;
    let sm_count = device.multiprocessor_count();
    let first = FIXED_SM120_TF32_BD_PAIRED_COMPARISONS[0];
    let first_emitter = FixedSm120Tf32BdComparisonEmitter::new(&sink);
    if device_cc != (12, 0) || sm_count != 170 {
        fixed_sm120_tf32_bd_reject(
            first,
            "none",
            device_cc,
            sm_count,
            "device_identity",
            format!(
                "requires physical CC12.0 with 170 SMs, found CC{}.{} with {} SMs",
                device_cc.0, device_cc.1, sm_count
            ),
            &first_emitter,
        );
    }
    let ctx = GpuCtx::new(&device).expect("GPU context");
    configure_fixed_auto_vendor_custom(&ctx, F32TriadPolicy::AllowDeterministicTf32);
    assert!(ctx.tf32(), "fast cuBLAS TF32 must remain enabled");

    for comparison in FIXED_SM120_TF32_BD_PAIRED_COMPARISONS {
        let emitter = FixedSm120Tf32BdComparisonEmitter::new(&sink);
        let shape = comparison.shape;
        let output_len = shape.m * shape.n;
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, comparison.dtype)
            .expect("TF32 B/D A allocation");
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, comparison.dtype)
            .expect("TF32 B/D B allocation");
        let bias = DtypedBuf::zeros(&ctx.stream, shape.n, comparison.dtype)
            .expect("TF32 B/D bias allocation");
        let incumbent_output = DtypedBuf::zeros(&ctx.stream, output_len, comparison.dtype)
            .expect("TF32 B/D incumbent output allocation");
        let candidate_output = DtypedBuf::zeros(&ctx.stream, output_len, comparison.dtype)
            .expect("TF32 B/D candidate output allocation");
        let vendor_output = DtypedBuf::zeros(&ctx.stream, output_len, comparison.dtype)
            .expect("TF32 B/D vendor output allocation");
        a.upload_f32(
            &ctx.stream,
            &synth(shape.m * shape.k, FIXED_SM120_TF32_BD_A_SEED),
        )
        .expect("TF32 B/D A upload");
        b.upload_f32(
            &ctx.stream,
            &synth(shape.k * shape.n, FIXED_SM120_TF32_BD_B_SEED),
        )
        .expect("TF32 B/D B upload");
        bias.upload_f32(&ctx.stream, &synth(shape.n, FIXED_SM120_TF32_BD_BIAS_SEED))
            .expect("TF32 B/D bias upload");
        let incumbent_operands = InferenceFwdOperands {
            c: typed(&incumbent_output, comparison.dtype),
            x: typed(&a, comparison.dtype),
            w: typed(&b, comparison.dtype),
            bias_ptr: None,
        };
        let candidate_operands = InferenceFwdOperands {
            c: typed(&candidate_output, comparison.dtype),
            ..incumbent_operands
        };
        let vendor_operands = InferenceFwdOperands {
            c: typed(&vendor_output, comparison.dtype),
            ..incumbent_operands
        };

        let mut expected_portable_rejection = false;
        for (arm, tile, operands) in [
            ("incumbent", comparison.incumbent, incumbent_operands),
            ("candidate", comparison.candidate, candidate_operands),
        ] {
            configure_fixed_auto_vendor_custom(&ctx, F32TriadPolicy::AllowDeterministicTf32);
            if let Err(error) = inference_forward_with_tile(&ctx, operands, shape, tile) {
                fixed_sm120_tf32_bd_reject(
                    comparison,
                    "none",
                    device_cc,
                    sm_count,
                    "direct_exec_preflight",
                    format!("{arm} direct Fixed enqueue failed: {error}"),
                    &emitter,
                );
            }
            if let Err(error) = ctx.stream.synchronize() {
                fixed_sm120_tf32_bd_reject(
                    comparison,
                    "none",
                    device_cc,
                    sm_count,
                    "direct_exec_preflight",
                    format!("{arm} direct Fixed synchronization failed: {error:?}"),
                    &emitter,
                );
            }
            let resource =
                fixed_sm120_tf32_bd_resource_snapshot(&ctx, shape, tile).unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        "none",
                        device_cc,
                        sm_count,
                        "resource_snapshot",
                        error,
                        &emitter,
                    )
                });
            match fixed_sm120_tf32_bd_resource_disposition(comparison, arm, tile, resource) {
                FixedSm120Tf32BdResourceDisposition::Pass => emitter.emit(
                    "preflight",
                    format!(
                        "{{{},\"record_type\":\"preflight\",\"arm\":\"{}\",\"execution\":\"direct_fixed_forward_with_tile\",\"preflight_phase\":\"before_warmup\",\"symbol\":\"{}\",\"grid\":{},\"block_threads\":{},\"dynamic_shared_bytes\":{},\"static_shared_bytes\":{},\"registers\":{},\"local_bytes\":{},\"occupancy_blocks_per_sm\":{},\"preferred_shared_memory_carveout\":{},\"passed\":true}}",
                        fixed_sm120_tf32_bd_common_json(comparison, "none", device_cc, sm_count, 0, 0),
                        arm,
                        resource.spec.symbol,
                        resource.grid,
                        resource.spec.block_threads,
                        resource.spec.dynamic_shared_bytes,
                        resource.static_shared_bytes,
                        resource.registers,
                        resource.local_bytes,
                        resource.occupancy_blocks_per_sm,
                        resource.preferred_shared_memory_carveout,
                    ),
                ),
                FixedSm120Tf32BdResourceDisposition::ExpectedPortableRegisterRejection => {
                    emitter.emit(
                        "rejection",
                        format!(
                            "{{{},\"record_type\":\"rejection\",\"failed_gate\":\"{}\",\"reason\":\"candidate {} uses {} registers, limit is 128\",\"expected_rejection\":true,\"promotion_eligible\":false}}",
                            fixed_sm120_tf32_bd_common_json(comparison, "none", device_cc, sm_count, 0, 0),
                            FIXED_SM120_TF32_BD_EXPECTED_PORTABLE_REJECTION_GATE,
                            resource.spec.symbol,
                            resource.registers,
                        ),
                    );
                    expected_portable_rejection = true;
                    break;
                }
                FixedSm120Tf32BdResourceDisposition::UnexpectedResourceDefect {
                    failed_gate,
                    reason,
                } => fixed_sm120_tf32_bd_reject(
                    comparison,
                    "none",
                    device_cc,
                    sm_count,
                    failed_gate,
                    reason,
                    &emitter,
                ),
            }
        }
        if expected_portable_rejection {
            assert_eq!(comparison.id, "d_portable_m64_vs_sm120_m64");
            assert!(
                emitter.counts.has_no_measurement_records(),
                "expected portable rejection must not emit bit, window, or summary records"
            );
            assert_eq!(
                emitter.counts.rejection.get(),
                1,
                "expected portable rejection must emit exactly one rejection"
            );
            continue;
        }

        for bias_name in FIXED_SM120_TF32_BD_BIASES {
            let bias_ptr = match bias_name {
                "none" => None,
                "synthesized" => Some(bias.cached_ptr()),
                _ => unreachable!("frozen bias inventory"),
            };
            let record = FixedSm120Tf32BdRecordContext {
                comparison,
                bias: bias_name,
                device_cc,
                sm_count,
                emitter: &emitter,
            };
            let incumbent_with_bias = InferenceFwdOperands {
                bias_ptr,
                ..incumbent_operands
            };
            let candidate_with_bias = InferenceFwdOperands {
                bias_ptr,
                ..candidate_operands
            };
            configure_fixed_auto_vendor_custom(&ctx, F32TriadPolicy::AllowDeterministicTf32);
            inference_forward_with_tile(&ctx, incumbent_with_bias, shape, comparison.incumbent)
                .unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "eager_bit_gate",
                        format!("incumbent eager enqueue failed: {error}"),
                        &emitter,
                    )
                });
            ctx.stream.synchronize().unwrap_or_else(|error| {
                fixed_sm120_tf32_bd_reject(
                    comparison,
                    bias_name,
                    device_cc,
                    sm_count,
                    "eager_bit_gate",
                    format!("incumbent eager synchronization failed: {error:?}"),
                    &emitter,
                )
            });
            let incumbent_bits = f32_bits(&ctx, &incumbent_output, output_len);
            let incumbent_digest = fixed_sm120_tf32_bd_digest(&incumbent_bits);
            fixed_sm120_tf32_bd_emit_bit_gate(
                record,
                "incumbent",
                "eager",
                None,
                &incumbent_digest,
                &incumbent_digest,
            );

            let mut candidate_bits = Vec::new();
            for _ in 0..2 {
                inference_forward_with_tile(&ctx, candidate_with_bias, shape, comparison.candidate)
                    .unwrap_or_else(|error| {
                        fixed_sm120_tf32_bd_reject(
                            comparison,
                            bias_name,
                            device_cc,
                            sm_count,
                            "eager_bit_gate",
                            format!("candidate eager enqueue failed: {error}"),
                            &emitter,
                        )
                    });
                ctx.stream.synchronize().unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "eager_bit_gate",
                        format!("candidate eager synchronization failed: {error:?}"),
                        &emitter,
                    )
                });
                candidate_bits = f32_bits(&ctx, &candidate_output, output_len);
                if candidate_bits != incumbent_bits {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "eager_bit_gate",
                        "candidate bits differ from incumbent bits",
                        &emitter,
                    );
                }
                let candidate_digest = fixed_sm120_tf32_bd_digest(&candidate_bits);
                if candidate_digest != incumbent_digest {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "eager_digest_gate",
                        "candidate digest differs from incumbent digest",
                        &emitter,
                    );
                }
                fixed_sm120_tf32_bd_emit_bit_gate(
                    record,
                    "candidate",
                    "eager",
                    None,
                    &candidate_digest,
                    &incumbent_digest,
                );
            }

            let candidate_run = || {
                inference_forward_with_tile(&ctx, candidate_with_bias, shape, comparison.candidate)
            };
            let candidate_graph = unsafe { capture_into_graph(&ctx.stream, candidate_run) }
                .unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_capture_gate",
                        format!("candidate graph capture failed: {error}"),
                        &emitter,
                    )
                });
            for replay_index in 0..10 {
                candidate_graph.launch().unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_replay_gate",
                        format!("candidate replay {replay_index} launch failed: {error:?}"),
                        &emitter,
                    )
                });
                ctx.stream.synchronize().unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_replay_gate",
                        format!("candidate replay {replay_index} sync failed: {error:?}"),
                        &emitter,
                    )
                });
                let replay_bits = f32_bits(&ctx, &candidate_output, output_len);
                if replay_bits != candidate_bits {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_replay_gate",
                        format!("candidate replay {replay_index} changed output bits"),
                        &emitter,
                    );
                }
                fixed_sm120_tf32_bd_emit_bit_gate(
                    record,
                    "candidate",
                    "graph_replay",
                    Some(replay_index),
                    &fixed_sm120_tf32_bd_digest(&replay_bits),
                    &fixed_sm120_tf32_bd_digest(&candidate_bits),
                );
            }

            let incumbent_run = || {
                inference_forward_with_tile(&ctx, incumbent_with_bias, shape, comparison.incumbent)
            };
            let incumbent_graph = unsafe { capture_into_graph(&ctx.stream, incumbent_run) }
                .unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_capture_gate",
                        format!("incumbent graph capture failed: {error}"),
                        &emitter,
                    )
                });
            for replay_index in 0..10 {
                incumbent_graph.launch().unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_replay_gate",
                        format!("incumbent replay {replay_index} launch failed: {error:?}"),
                        &emitter,
                    )
                });
                ctx.stream.synchronize().unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_replay_gate",
                        format!("incumbent replay {replay_index} sync failed: {error:?}"),
                        &emitter,
                    )
                });
                let replay_bits = f32_bits(&ctx, &incumbent_output, output_len);
                if replay_bits != incumbent_bits {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "graph_replay_gate",
                        format!("incumbent replay {replay_index} changed output bits"),
                        &emitter,
                    );
                }
                fixed_sm120_tf32_bd_emit_bit_gate(
                    record,
                    "incumbent",
                    "graph_replay",
                    Some(replay_index),
                    &fixed_sm120_tf32_bd_digest(&replay_bits),
                    &incumbent_digest,
                );
            }

            let prefix_shape = InferenceShape {
                m: 1,
                k: shape.k,
                n: shape.n,
            };
            for (arm, tile, full_bits) in [
                ("incumbent", comparison.incumbent, &incumbent_bits),
                ("candidate", comparison.candidate, &candidate_bits),
            ] {
                let prefix_output = DtypedBuf::zeros(&ctx.stream, shape.n, comparison.dtype)
                    .expect("TF32 B/D prefix output allocation");
                let prefix_operands = InferenceFwdOperands {
                    c: typed(&prefix_output, comparison.dtype),
                    x: typed(&a, comparison.dtype),
                    w: typed(&b, comparison.dtype),
                    bias_ptr,
                };
                inference_forward_with_tile(&ctx, prefix_operands, prefix_shape, tile)
                    .unwrap_or_else(|error| {
                        fixed_sm120_tf32_bd_reject(
                            comparison,
                            bias_name,
                            device_cc,
                            sm_count,
                            "prefix_gate",
                            format!("{arm} prefix enqueue failed: {error}"),
                            &emitter,
                        )
                    });
                ctx.stream.synchronize().unwrap_or_else(|error| {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "prefix_gate",
                        format!("{arm} prefix synchronization failed: {error:?}"),
                        &emitter,
                    )
                });
                let prefix_bits = f32_bits(&ctx, &prefix_output, shape.n);
                let reference_row = &full_bits[..shape.n];
                if prefix_bits != reference_row {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        bias_name,
                        device_cc,
                        sm_count,
                        "prefix_gate",
                        format!("{arm} leading m=1 output differs from full row zero"),
                        &emitter,
                    );
                }
                fixed_sm120_tf32_bd_emit_bit_gate(
                    record,
                    arm,
                    "prefix",
                    None,
                    &fixed_sm120_tf32_bd_digest(&prefix_bits),
                    &fixed_sm120_tf32_bd_digest(reference_row),
                );
            }
        }

        configure_fixed_auto_vendor_custom(&ctx, F32TriadPolicy::AllowDeterministicTf32);
        for _ in 0..FIXED_SM120_TF32_BD_WARMUPS {
            inference_forward_with_tile(&ctx, incumbent_operands, shape, comparison.incumbent)
                .expect("TF32 B/D incumbent warmup");
        }
        ctx.stream
            .synchronize()
            .expect("TF32 B/D incumbent warmup sync");
        configure_fixed_auto_vendor_custom(&ctx, F32TriadPolicy::AllowDeterministicTf32);
        for _ in 0..FIXED_SM120_TF32_BD_WARMUPS {
            inference_forward_with_tile(&ctx, candidate_operands, shape, comparison.candidate)
                .expect("TF32 B/D candidate warmup");
        }
        ctx.stream
            .synchronize()
            .expect("TF32 B/D candidate warmup sync");
        configure_fixed_auto_vendor_vendor(&ctx, F32TriadPolicy::AllowDeterministicTf32);
        for _ in 0..FIXED_SM120_TF32_BD_WARMUPS {
            launch_fixed_auto_vendor_vendor(&ctx, vendor_operands, shape);
        }
        ctx.stream
            .synchronize()
            .expect("TF32 B/D vendor warmup sync");

        let incumbent_iterations = fixed_sm120_tf32_bd_forced_iterations(
            &ctx,
            incumbent_operands,
            shape,
            comparison.incumbent,
        );
        let candidate_iterations = fixed_sm120_tf32_bd_forced_iterations(
            &ctx,
            candidate_operands,
            shape,
            comparison.candidate,
        );
        let vendor_pilot_us = fixed_auto_vendor_vendor_window_us(
            &ctx,
            vendor_operands,
            shape,
            F32TriadPolicy::AllowDeterministicTf32,
            FIXED_SM120_TF32_BD_PILOT_ITERATIONS,
        );
        let vendor_iterations = fixed_auto_vendor_iterations(vendor_pilot_us);

        let mut window_records = Vec::with_capacity(4 * FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
        let mut summaries = Vec::with_capacity(4);
        for comparator_order in FIXED_SM120_TF32_BD_ROUTE_ORDERS {
            fixed_sm120_tf32_bd_environment_preflight(&format!(
                "{} {comparator_order}",
                comparison.id
            ))
            .unwrap_or_else(|error| {
                fixed_sm120_tf32_bd_reject(
                    comparison,
                    "none",
                    device_cc,
                    sm_count,
                    "environment_preflight",
                    error,
                    &emitter,
                )
            });
            let candidate_first = comparator_order == "candidate_then_incumbent";
            let mut candidate_times = Vec::with_capacity(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
            let mut incumbent_times = Vec::with_capacity(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
            let mut ratios = Vec::with_capacity(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
            for pair_index in 0..FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER {
                let (candidate_us, incumbent_us) = if candidate_first {
                    (
                        fixed_sm120_tf32_bd_forced_window_us(
                            &ctx,
                            candidate_operands,
                            shape,
                            comparison.candidate,
                            candidate_iterations,
                        ),
                        fixed_sm120_tf32_bd_forced_window_us(
                            &ctx,
                            incumbent_operands,
                            shape,
                            comparison.incumbent,
                            incumbent_iterations,
                        ),
                    )
                } else {
                    let incumbent_us = fixed_sm120_tf32_bd_forced_window_us(
                        &ctx,
                        incumbent_operands,
                        shape,
                        comparison.incumbent,
                        incumbent_iterations,
                    );
                    let candidate_us = fixed_sm120_tf32_bd_forced_window_us(
                        &ctx,
                        candidate_operands,
                        shape,
                        comparison.candidate,
                        candidate_iterations,
                    );
                    (candidate_us, incumbent_us)
                };
                if !candidate_us.is_finite()
                    || candidate_us <= 0.0
                    || !incumbent_us.is_finite()
                    || incumbent_us <= 0.0
                {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        "none",
                        device_cc,
                        sm_count,
                        "timing_value",
                        format!(
                            "non-positive or non-finite route timing candidate={candidate_us} incumbent={incumbent_us}"
                        ),
                        &emitter,
                    );
                }
                let ratio = candidate_us / incumbent_us;
                candidate_times.push(candidate_us);
                incumbent_times.push(incumbent_us);
                ratios.push(ratio);
                window_records.push(format!(
                    "{{{},\"record_type\":\"window\",\"comparison_kind\":\"candidate_vs_incumbent\",\"comparator_order\":\"{}\",\"pair_index\":{},\"first_arm\":\"{}\",\"second_arm\":\"{}\",\"candidate_us\":{:.9},\"comparator_us\":{:.9},\"ratio\":{:.9},\"candidate_iterations\":{},\"comparator_iterations\":{}}}",
                    fixed_sm120_tf32_bd_common_json(comparison, "none", device_cc, sm_count, incumbent_iterations, candidate_iterations),
                    comparator_order,
                    pair_index,
                    if candidate_first { "candidate" } else { "incumbent" },
                    if candidate_first { "incumbent" } else { "candidate" },
                    candidate_us,
                    incumbent_us,
                    ratio,
                    candidate_iterations,
                    incumbent_iterations,
                ));
            }
            summaries.push(fixed_sm120_tf32_bd_summary(
                "candidate_vs_incumbent",
                comparator_order,
                &candidate_times,
                &incumbent_times,
                &ratios,
            ));
        }

        for comparator_order in FIXED_SM120_TF32_BD_VENDOR_ORDERS {
            fixed_sm120_tf32_bd_environment_preflight(&format!(
                "{} {comparator_order}",
                comparison.id
            ))
            .unwrap_or_else(|error| {
                fixed_sm120_tf32_bd_reject(
                    comparison,
                    "none",
                    device_cc,
                    sm_count,
                    "environment_preflight",
                    error,
                    &emitter,
                )
            });
            let candidate_first = comparator_order == "candidate_then_vendor";
            let mut candidate_times = Vec::with_capacity(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
            let mut vendor_times = Vec::with_capacity(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
            let mut ratios = Vec::with_capacity(FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER);
            for pair_index in 0..FIXED_SM120_TF32_BD_WINDOWS_PER_ORDER {
                let (candidate_us, vendor_us) = if candidate_first {
                    (
                        fixed_sm120_tf32_bd_forced_window_us(
                            &ctx,
                            candidate_operands,
                            shape,
                            comparison.candidate,
                            candidate_iterations,
                        ),
                        fixed_auto_vendor_vendor_window_us(
                            &ctx,
                            vendor_operands,
                            shape,
                            F32TriadPolicy::AllowDeterministicTf32,
                            vendor_iterations,
                        ),
                    )
                } else {
                    let vendor_us = fixed_auto_vendor_vendor_window_us(
                        &ctx,
                        vendor_operands,
                        shape,
                        F32TriadPolicy::AllowDeterministicTf32,
                        vendor_iterations,
                    );
                    let candidate_us = fixed_sm120_tf32_bd_forced_window_us(
                        &ctx,
                        candidate_operands,
                        shape,
                        comparison.candidate,
                        candidate_iterations,
                    );
                    (candidate_us, vendor_us)
                };
                if !candidate_us.is_finite()
                    || candidate_us <= 0.0
                    || !vendor_us.is_finite()
                    || vendor_us <= 0.0
                {
                    fixed_sm120_tf32_bd_reject(
                        comparison,
                        "none",
                        device_cc,
                        sm_count,
                        "timing_value",
                        format!(
                            "non-positive or non-finite vendor timing candidate={candidate_us} vendor={vendor_us}"
                        ),
                        &emitter,
                    );
                }
                let ratio = candidate_us / vendor_us;
                candidate_times.push(candidate_us);
                vendor_times.push(vendor_us);
                ratios.push(ratio);
                window_records.push(format!(
                    "{{{},\"record_type\":\"window\",\"comparison_kind\":\"candidate_vs_vendor\",\"comparator_order\":\"{}\",\"pair_index\":{},\"first_arm\":\"{}\",\"second_arm\":\"{}\",\"candidate_us\":{:.9},\"comparator_us\":{:.9},\"ratio\":{:.9},\"candidate_iterations\":{},\"comparator_iterations\":{},\"comparator_tile\":\"fast_cublas_tf32_allowed\"}}",
                    fixed_sm120_tf32_bd_common_json(comparison, "none", device_cc, sm_count, incumbent_iterations, candidate_iterations),
                    comparator_order,
                    pair_index,
                    if candidate_first { "candidate" } else { "vendor" },
                    if candidate_first { "vendor" } else { "candidate" },
                    candidate_us,
                    vendor_us,
                    ratio,
                    candidate_iterations,
                    vendor_iterations,
                ));
            }
            summaries.push(fixed_sm120_tf32_bd_summary(
                "candidate_vs_vendor",
                comparator_order,
                &candidate_times,
                &vendor_times,
                &ratios,
            ));
        }

        let route_summaries = &summaries[..2];
        let vendor_summaries = &summaries[2..];
        let route_order_gate = route_summaries
            .iter()
            .all(|summary| summary.ratio_p50 <= 0.995 && summary.ratio_p95 < 1.0);
        let vendor_order_gate = vendor_summaries
            .iter()
            .all(|summary| summary.ratio_p50 <= 1.0 && summary.ratio_p95 < 1.0);
        let gap_close_gate = route_summaries
            .iter()
            .all(|summary| summary.ratio_p50 <= comparison.incumbent_gap_close_ratio);
        let is_m128_candidate = comparison.candidate == InferenceTile::Tf32Sm120M128S2;
        let promotion_eligible =
            is_m128_candidate && route_order_gate && vendor_order_gate && gap_close_gate;
        let vendor_verdict = if vendor_summaries
            .iter()
            .all(|summary| summary.ratio_p50 <= 0.995 && summary.ratio_p95 < 1.0)
        {
            "clear_vendor_win"
        } else if vendor_order_gate {
            "parity_admissible"
        } else {
            "rejected"
        };

        for record in window_records {
            emitter.emit("window", record);
        }
        for summary in &summaries {
            let comparator_tile = if summary.comparison_kind == "candidate_vs_vendor" {
                ",\"comparator_tile\":\"fast_cublas_tf32_allowed\""
            } else {
                ""
            };
            let verdict = if summary.comparison_kind == "candidate_vs_vendor" {
                format!(",\"vendor_verdict\":\"{vendor_verdict}\"")
            } else {
                String::new()
            };
            emitter.emit(
                "summary",
                format!(
                    "{{{},\"record_type\":\"summary\",\"comparison_kind\":\"{}\",\"comparator_order\":\"{}\",\"candidate_p05_us\":{:.9},\"candidate_p50_us\":{:.9},\"candidate_p95_us\":{:.9},\"comparator_p05_us\":{:.9},\"comparator_p50_us\":{:.9},\"comparator_p95_us\":{:.9},\"ratio_p05\":{:.9},\"ratio_p50\":{:.9},\"ratio_p95\":{:.9},\"direct_preflight_passed\":true,\"resource_gate_passed\":true,\"exact_bits_passed\":true,\"graph_replay_passed\":true,\"prefix_gate_passed\":true,\"environment_preflight_passed\":true,\"promotion_eligible\":{}{}{}}}",
                    fixed_sm120_tf32_bd_common_json(
                        comparison,
                        "none",
                        device_cc,
                        sm_count,
                        incumbent_iterations,
                        candidate_iterations
                    ),
                    summary.comparison_kind,
                    summary.comparator_order,
                    summary.candidate_p05_us,
                    summary.candidate_p50_us,
                    summary.candidate_p95_us,
                    summary.comparator_p05_us,
                    summary.comparator_p50_us,
                    summary.comparator_p95_us,
                    summary.ratio_p05,
                    summary.ratio_p50,
                    summary.ratio_p95,
                    promotion_eligible,
                    comparator_tile,
                    verdict,
                ),
            );
        }
        if is_m128_candidate && !promotion_eligible {
            emitter.emit(
                "rejection",
                format!(
                    "{{{},\"record_type\":\"rejection\",\"failed_gate\":\"strict_decision_rule\",\"reason\":\"one or more paired order or vendor-gap thresholds failed\",\"route_order_gate\":{},\"vendor_order_gate\":{},\"gap_close_gate\":{},\"promotion_eligible\":false}}",
                    fixed_sm120_tf32_bd_common_json(
                        comparison,
                        "none",
                        device_cc,
                        sm_count,
                        incumbent_iterations,
                        candidate_iterations
                    ),
                    route_order_gate,
                    vendor_order_gate,
                    gap_close_gate,
                ),
            );
        }
    }
}

#[derive(Clone, Copy)]
struct FixedAutoVendorCell {
    label: &'static str,
    shape: InferenceShape,
    expected: InferenceTile,
}

#[derive(Clone, Copy)]
struct FixedAutoVendorRow {
    name: &'static str,
    input_dtype: WeightDtype,
    output_dtype: WeightDtype,
    policy: F32TriadPolicy,
    policy_name: &'static str,
    cells: &'static [FixedAutoVendorCell],
}

const FIXED_AUTO_VENDOR_HALF_CELLS: &[FixedAutoVendorCell] = &[
    FixedAutoVendorCell {
        label: "hot_a",
        shape: InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "hot_b",
        shape: InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "hot_c",
        shape: InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "hot_d",
        shape: InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "hot_e",
        shape: InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m512_k1928_n2304",
        shape: InferenceShape {
            m: 512,
            k: 1928,
            n: 2304,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1024_k1928_n1536",
        shape: InferenceShape {
            m: 1024,
            k: 1928,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1024_k1928_n1928",
        shape: InferenceShape {
            m: 1024,
            k: 1928,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1024_k1928_n2304",
        shape: InferenceShape {
            m: 1024,
            k: 1928,
            n: 2304,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1536_k1928_n1536",
        shape: InferenceShape {
            m: 1536,
            k: 1928,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1536_k1928_n1928",
        shape: InferenceShape {
            m: 1536,
            k: 1928,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1536_k1928_n2304",
        shape: InferenceShape {
            m: 1536,
            k: 1928,
            n: 2304,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m2048_k1928_n1536",
        shape: InferenceShape {
            m: 2048,
            k: 1928,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m2048_k1928_n1928",
        shape: InferenceShape {
            m: 2048,
            k: 1928,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m2048_k1928_n2304",
        shape: InferenceShape {
            m: 2048,
            k: 1928,
            n: 2304,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m3072_k1928_n1536",
        shape: InferenceShape {
            m: 3072,
            k: 1928,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m3072_k1928_n1928",
        shape: InferenceShape {
            m: 3072,
            k: 1928,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "deep_m3072_k1928_n2304",
        shape: InferenceShape {
            m: 3072,
            k: 1928,
            n: 2304,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m4621_k1928_n1928",
        shape: InferenceShape {
            m: 4621,
            k: 1928,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m4621_k1928_n2304",
        shape: InferenceShape {
            m: 4621,
            k: 1928,
            n: 2304,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m1536_k1032_n1536",
        shape: InferenceShape {
            m: 1536,
            k: 1032,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2),
    },
    FixedAutoVendorCell {
        label: "deep_m2048_k1032_n1536",
        shape: InferenceShape {
            m: 2048,
            k: 1032,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m1536_k384_n1536",
        shape: InferenceShape {
            m: 1536,
            k: 384,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m1536_k768_n1536",
        shape: InferenceShape {
            m: 1536,
            k: 768,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m1536_k384_n1928",
        shape: InferenceShape {
            m: 1536,
            k: 384,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m2048_k384_n1928",
        shape: InferenceShape {
            m: 2048,
            k: 384,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m2048_k768_n1928",
        shape: InferenceShape {
            m: 2048,
            k: 768,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m3072_k384_n1928",
        shape: InferenceShape {
            m: 3072,
            k: 384,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k768_n1536",
        shape: InferenceShape {
            m: 4096,
            k: 768,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k384_n1928",
        shape: InferenceShape {
            m: 4096,
            k: 384,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k768_n1928",
        shape: InferenceShape {
            m: 4096,
            k: 768,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k768_n2304",
        shape: InferenceShape {
            m: 4096,
            k: 768,
            n: 2304,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m1536_k768_n1928",
        shape: InferenceShape {
            m: 1536,
            k: 768,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m2048_k384_n1536",
        shape: InferenceShape {
            m: 2048,
            k: 384,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m2048_k768_n1536",
        shape: InferenceShape {
            m: 2048,
            k: 768,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m3072_k768_n1928",
        shape: InferenceShape {
            m: 3072,
            k: 768,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k384_n1536",
        shape: InferenceShape {
            m: 4096,
            k: 384,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m4621_k384_n1536",
        shape: InferenceShape {
            m: 4621,
            k: 384,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m4621_k768_n1536",
        shape: InferenceShape {
            m: 4621,
            k: 768,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m3072_k768_n2304",
        shape: InferenceShape {
            m: 3072,
            k: 768,
            n: 2304,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m1536_k512_n1928",
        shape: InferenceShape {
            m: 1536,
            k: 512,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "shallow_m1536_k520_n1928",
        shape: InferenceShape {
            m: 1536,
            k: 520,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k512_n1536",
        shape: InferenceShape {
            m: 4096,
            k: 512,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "shallow_m4096_k520_n1536",
        shape: InferenceShape {
            m: 4096,
            k: 520,
            n: 1536,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
];

const FIXED_AUTO_VENDOR_TF32_CELLS: &[FixedAutoVendorCell] = &[
    FixedAutoVendorCell {
        label: "hot_a",
        shape: InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        expected: InferenceTile::Tf32Sm120M64S2,
    },
    FixedAutoVendorCell {
        label: "hot_b",
        shape: InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        expected: InferenceTile::Tf32Sm120M64S2,
    },
    FixedAutoVendorCell {
        label: "hot_c",
        shape: InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        expected: InferenceTile::Tf32Sm120M64S2,
    },
    FixedAutoVendorCell {
        label: "hot_d",
        shape: InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        expected: InferenceTile::Tf32Sm120M64S2,
    },
    FixedAutoVendorCell {
        label: "hot_e",
        shape: InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        expected: InferenceTile::Tf32Sm120M64S2,
    },
];

const FIXED_AUTO_VENDOR_MIXED_CELLS: &[FixedAutoVendorCell] = &[
    FixedAutoVendorCell {
        label: "hot_a",
        shape: InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    },
    FixedAutoVendorCell {
        label: "hot_b",
        shape: InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S3),
    },
    FixedAutoVendorCell {
        label: "hot_c",
        shape: InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "hot_d",
        shape: InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
    FixedAutoVendorCell {
        label: "hot_e",
        shape: InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        expected: InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2),
    },
];

const FIXED_AUTO_VENDOR_EXACT_CELLS: &[FixedAutoVendorCell] = &[
    FixedAutoVendorCell {
        label: "hot_a",
        shape: InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        expected: InferenceTile::F32N128S2,
    },
    FixedAutoVendorCell {
        label: "hot_b",
        shape: InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        expected: InferenceTile::F32N128S2,
    },
    FixedAutoVendorCell {
        label: "hot_c",
        shape: InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        expected: InferenceTile::Legacy,
    },
    FixedAutoVendorCell {
        label: "hot_d",
        shape: InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        expected: InferenceTile::Legacy,
    },
    FixedAutoVendorCell {
        label: "hot_e",
        shape: InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        expected: InferenceTile::Legacy,
    },
];

fn expected_ada_half_auto_before_streamk(
    nvrtc: (i32, i32),
    dtype: WeightDtype,
    shape: InferenceShape,
    has_bias: bool,
) -> Option<InferenceTile> {
    use InferenceTile::{
        Tc128Sm89Pipeline as Pipeline, Tc128Sm89S3 as S3, Tc128Sm89Swizzle as Swizzle,
    };

    match (nvrtc, dtype, (shape.m, shape.k, shape.n), has_bias) {
        ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 384, 1928), false) => Some(Pipeline),
        ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 384, 1928), true) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 768, 2304), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 1928, 384), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::Bf16, (2048, 768, 2304), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::Bf16, (2048, 2304, 768), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::F16, (4621, 384, 1928), _) => Some(Pipeline),
        ((12, 8) | (13, 0), WeightDtype::F16, (4621, 768, 2304), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::F16, (4621, 1928, 384), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::F16, (2048, 768, 2304), _) => Some(Swizzle),
        ((12, 8) | (13, 0), WeightDtype::F16, (2048, 2304, 768), _) => Some(Swizzle),
        ((13, 2), WeightDtype::Bf16, (4621, 384, 1928), _) => Some(Pipeline),
        ((13, 2), WeightDtype::Bf16, (4621, 768, 2304), false) => Some(S3),
        ((13, 2), WeightDtype::Bf16, (4621, 768, 2304), true) => Some(Swizzle),
        ((13, 2), WeightDtype::Bf16, (4621, 1928, 384), _) => Some(Pipeline),
        ((13, 2), WeightDtype::Bf16, (2048, 768, 2304), _) => Some(Swizzle),
        ((13, 2), WeightDtype::Bf16, (2048, 2304, 768), _) => Some(Swizzle),
        ((13, 2), WeightDtype::F16, (4621, 384, 1928), _) => Some(Pipeline),
        ((13, 2), WeightDtype::F16, (4621, 768, 2304), false) => Some(S3),
        ((13, 2), WeightDtype::F16, (4621, 768, 2304), true) => Some(Swizzle),
        ((13, 2), WeightDtype::F16, (4621, 1928, 384), _) => Some(Pipeline),
        ((13, 2), WeightDtype::F16, (2048, 768, 2304), _) => Some(Swizzle),
        ((13, 2), WeightDtype::F16, (2048, 2304, 768), _) => Some(Pipeline),
        _ => None,
    }
}

#[test]
fn ada_half_auto_harness_expectation_is_literal_and_fail_closed() {
    for &nvrtc in &[(12, 8), (13, 0), (13, 2)] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for cell in FIXED_AUTO_VENDOR_EXACT_CELLS {
                for has_bias in [false, true] {
                    assert!(
                        expected_ada_half_auto_before_streamk(nvrtc, dtype, cell.shape, has_bias)
                            .is_some(),
                        "{nvrtc:?} {dtype:?} {} bias={has_bias}",
                        cell.label
                    );
                }
            }
        }
    }
    let hot_a = FIXED_AUTO_VENDOR_EXACT_CELLS[0].shape;
    assert_eq!(
        expected_ada_half_auto_before_streamk((12, 8), WeightDtype::Bf16, hot_a, false),
        Some(InferenceTile::Tc128Sm89Pipeline)
    );
    assert_eq!(
        expected_ada_half_auto_before_streamk((12, 8), WeightDtype::Bf16, hot_a, true),
        Some(InferenceTile::Tc128Sm89Swizzle)
    );
    for nvrtc in [(12, 7), (13, 1), (13, 3), (14, 0)] {
        assert_eq!(
            expected_ada_half_auto_before_streamk(nvrtc, WeightDtype::Bf16, hot_a, false),
            None
        );
    }
    assert_eq!(
        expected_ada_half_auto_before_streamk((13, 2), WeightDtype::F32, hot_a, false),
        None
    );
    assert_eq!(
        expected_ada_half_auto_before_streamk(
            (13, 2),
            WeightDtype::F16,
            InferenceShape {
                m: hot_a.m - 1,
                ..hot_a
            },
            false
        ),
        None
    );
    let hot_b = FIXED_AUTO_VENDOR_EXACT_CELLS[1].shape;
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        assert_eq!(
            expected_ada_half_auto_before_streamk((13, 2), dtype, hot_b, false),
            Some(InferenceTile::Tc128Sm89S3)
        );
        assert_eq!(
            expected_ada_half_auto_before_streamk((13, 2), dtype, hot_b, true),
            Some(InferenceTile::Tc128Sm89Swizzle)
        );
    }
}

// Exactly the five rows admitted by the frozen production forced21 batch.
// This is a test oracle, not the production selector or its helper.
fn expected_ada_finalist_auto(
    nvrtc: (i32, i32),
    row: &str,
    shape: InferenceShape,
    has_bias: bool,
) -> Option<InferenceTile> {
    match (nvrtc, row, (shape.m, shape.k, shape.n), has_bias) {
        ((12, 8) | (13, 0) | (13, 2), "tf32", (2048, 2304, 768), false) => {
            Some(InferenceTile::Tf32RnaM128N96S3)
        }
        ((13, 2), "f16", (2048, 768, 2304), false) => Some(InferenceTile::TcM64N64Sm89S3),
        ((13, 2), "f16", (2048, 2304, 768), false) => Some(InferenceTile::TcM128N64Sm89S2),
        _ => None,
    }
}

fn expected_ada_half_auto(
    nvrtc: (i32, i32),
    dtype: WeightDtype,
    shape: InferenceShape,
    has_bias: bool,
) -> Option<InferenceTile> {
    expected_ada_finalist_auto(nvrtc, dtype.as_str(), shape, has_bias)
        .or_else(|| expected_ada_half_auto_before_streamk(nvrtc, dtype, shape, has_bias))
}

#[test]
fn ada_finalist_auto_harness_oracle_has_exactly_five_promotions() {
    let mut promotions = 0;
    for nvrtc in [(12, 7), (12, 8), (13, 0), (13, 1), (13, 2), (13, 3)] {
        for row in ["tf32", "f16", "bf16", "f16_f32", "f32_exact_fast"] {
            for (index, cell) in FIXED_AUTO_VENDOR_EXACT_CELLS.iter().copied().enumerate() {
                for has_bias in [false, true] {
                    let want = match (nvrtc, row, index, has_bias) {
                        ((12, 8) | (13, 0) | (13, 2), "tf32", 4, false) => {
                            Some(InferenceTile::Tf32RnaM128N96S3)
                        }
                        ((13, 2), "f16", 3, false) => Some(InferenceTile::TcM64N64Sm89S3),
                        ((13, 2), "f16", 4, false) => Some(InferenceTile::TcM128N64Sm89S2),
                        _ => None,
                    };
                    assert_eq!(
                        expected_ada_finalist_auto(nvrtc, row, cell.shape, has_bias),
                        want
                    );
                    promotions += usize::from(want.is_some());
                    for shape in [
                        InferenceShape {
                            m: cell.shape.m - 1,
                            ..cell.shape
                        },
                        InferenceShape {
                            k: cell.shape.k - 1,
                            ..cell.shape
                        },
                        InferenceShape {
                            n: cell.shape.n - 1,
                            ..cell.shape
                        },
                    ] {
                        assert_eq!(
                            expected_ada_finalist_auto(nvrtc, row, shape, has_bias),
                            None
                        );
                    }
                }
            }
        }
    }
    assert_eq!(promotions, 5);
    for nvrtc in [(12, 8), (13, 0), (13, 2)] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for (index, cell) in FIXED_AUTO_VENDOR_EXACT_CELLS.iter().copied().enumerate() {
                for bias in [false, true] {
                    let old = expected_ada_half_auto_before_streamk(nvrtc, dtype, cell.shape, bias);
                    let want = match (nvrtc, dtype, index, bias) {
                        ((13, 2), WeightDtype::F16, 3, false) => {
                            Some(InferenceTile::TcM64N64Sm89S3)
                        }
                        ((13, 2), WeightDtype::F16, 4, false) => {
                            Some(InferenceTile::TcM128N64Sm89S2)
                        }
                        _ => old,
                    };
                    assert_eq!(expected_ada_half_auto(nvrtc, dtype, cell.shape, bias), want);
                }
            }
        }
    }
}

fn fixed_auto_vendor_expected_exact_tile(
    cell: FixedAutoVendorCell,
    device_cc: (u32, u32),
    sm_count: u32,
    nvrtc_version: (i32, i32),
    nvrtc_library_known: bool,
    has_bias: bool,
    t256_loaded: bool,
) -> InferenceTile {
    match device_cc {
        (12, 0) if sm_count == 170 && nvrtc_version == (13, 2) && nvrtc_library_known => {
            match ((cell.shape.m, cell.shape.k, cell.shape.n), has_bias) {
                ((4621, 384, 1928), false) => InferenceTile::F32Sm120TmaFmaM64N128,
                ((4621, 384, 1928), true) if t256_loaded => {
                    InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
                }
                ((4621, 384, 1928), true) => InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
                ((4621, 768, 2304), false) => InferenceTile::F32Sm120TmaFmaM128N64,
                ((4621, 768, 2304), true) => InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
                ((4621, 1928, 384), true) => InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96,
                ((2048, 768, 2304), _) => InferenceTile::F32Sm120N64CopyPlan,
                ((2048, 2304, 768), _) => InferenceTile::F32Sm120N64CopyPlan,
                _ => cell.expected,
            }
        }
        (12, 0) => cell.expected,
        (12, 1) => InferenceTile::Legacy,
        _ => panic!(
            "the AUTO/vendor survey does not admit CC{}.{}",
            device_cc.0, device_cc.1
        ),
    }
}

fn fixed_auto_vendor_expected_half_tile(
    cell: FixedAutoVendorCell,
    dtype: WeightDtype,
    device_cc: (u32, u32),
    sm_count: u32,
    nvrtc_version: (i32, i32),
    has_bias: bool,
) -> InferenceTile {
    if device_cc == (12, 0)
        && sm_count == 170
        && nvrtc_version == (13, 2)
        && matches!(dtype, WeightDtype::Bf16 | WeightDtype::F16)
    {
        let dims = (cell.shape.m, cell.shape.k, cell.shape.n);
        let s3 = dims == (4621, 768, 2304)
            || (!has_bias
                && (matches!(dims, (2048, 1928, 2304) | (1536, 1032, 1536))
                    || (dtype == WeightDtype::F16
                        && matches!(
                            dims,
                            (1024, 1928, 1928)
                                | (1024, 1928, 2304)
                                | (1536, 1928, 1536)
                                | (2048, 768, 2304)
                                | (3072, 768, 2304)
                                | (4621, 768, 1536)
                        ))));
        if s3 {
            return InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S3);
        }
    }
    if device_cc == (12, 0)
        && sm_count == 170
        && nvrtc_version == (13, 2)
        && matches!(dtype, WeightDtype::Bf16 | WeightDtype::F16)
        && matches!(
            (cell.shape.m, cell.shape.k, cell.shape.n),
            (512, 1928, 2304)
                | (1536, 1928, 2304)
                | (1536, 1928, 1928)
                | (3072, 1928, 1928)
                | (4096, 520, 1536)
                | (2048, 1928, 1536)
                | (1024, 1928, 1928)
                | (1024, 1928, 2304)
                | (1536, 1032, 1536)
                | (1536, 1928, 1536)
                | (4621, 1928, 1928)
                | (1536, 768, 1536)
        )
    {
        return InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N128Bk64S2);
    }
    cell.expected
}

fn fixed_auto_vendor_expected_tf32_tile(
    cell: FixedAutoVendorCell,
    device_cc: (u32, u32),
    sm_count: u32,
    nvrtc_version: (i32, i32),
    has_bias: bool,
    output_aligned: bool,
    nvrtc_library_known: bool,
) -> InferenceTile {
    if device_cc == (12, 0) && sm_count == 170 && nvrtc_version == (13, 2) {
        match (cell.shape.m, cell.shape.k, cell.shape.n, has_bias) {
            (4621, 384, 1928, _) if output_aligned && nvrtc_library_known => {
                return InferenceTile::Tf32Sm120M128S2;
            }
            (4621, 768, 2304, _) => return InferenceTile::Tf32Sm120M128S2,
            (2048, 768, 2304, false) | (2048, 768, 2304, true) => {
                return if output_aligned {
                    InferenceTile::Tf32Sm120M64S2PairStore
                } else {
                    InferenceTile::Tf32Sm120M64S2ProducerWarp
                };
            }
            _ => {}
        }
    }
    cell.expected
}

fn fixed_auto_vendor_expected_mixed_tile(
    cell: FixedAutoVendorCell,
    device_cc: (u32, u32),
    sm_count: u32,
    nvrtc_version: (i32, i32),
) -> InferenceTile {
    if device_cc == (12, 0) && sm_count == 170 && nvrtc_version == (13, 2) {
        return cell.expected;
    }
    if cell.label == "hot_b" {
        InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2)
    } else {
        cell.expected
    }
}

#[test]
fn fixed_auto_vendor_exact_expectations_are_device_specific() {
    let hot_a = FIXED_AUTO_VENDOR_EXACT_CELLS[0];
    let hot_b = FIXED_AUTO_VENDOR_EXACT_CELLS[1];
    let hot_d = FIXED_AUTO_VENDOR_EXACT_CELLS[3];
    for bias in [false, true] {
        assert_eq!(
            fixed_auto_vendor_expected_exact_tile(hot_d, (12, 0), 170, (13, 2), true, bias, true),
            InferenceTile::F32Sm120N64CopyPlan
        );
        assert_eq!(
            fixed_auto_vendor_expected_exact_tile(hot_d, (12, 0), 169, (13, 2), true, bias, true),
            InferenceTile::Legacy
        );
    }
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_a, (12, 0), 170, (13, 2), true, false, true),
        InferenceTile::F32Sm120TmaFmaM64N128
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_a, (12, 0), 170, (13, 2), true, true, true),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_a, (12, 0), 170, (13, 2), true, true, false),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_b, (12, 0), 170, (13, 2), true, false, true),
        InferenceTile::F32Sm120TmaFmaM128N64
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_b, (12, 0), 170, (13, 2), true, true, true),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(
            FIXED_AUTO_VENDOR_EXACT_CELLS[2],
            (12, 0),
            170,
            (13, 2),
            true,
            true,
            true
        ),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(
            FIXED_AUTO_VENDOR_EXACT_CELLS[2],
            (12, 0),
            170,
            (13, 2),
            true,
            false,
            true
        ),
        InferenceTile::Legacy
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_a, (12, 0), 169, (13, 2), true, false, true),
        InferenceTile::F32N128S2
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_b, (12, 0), 170, (13, 2), false, false, true),
        InferenceTile::F32N128S2
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_b, (12, 0), 170, (13, 3), true, false, true),
        InferenceTile::F32N128S2
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_a, (12, 1), 170, (13, 2), true, false, true),
        InferenceTile::Legacy
    );
    assert_eq!(
        fixed_auto_vendor_expected_exact_tile(hot_b, (12, 1), 170, (13, 2), true, false, true),
        InferenceTile::Legacy
    );
}

#[test]
fn fixed_auto_vendor_tf32_expectation_tracks_the_qualified_compiler_cells() {
    let hot_a = FIXED_AUTO_VENDOR_TF32_CELLS[0];
    let hot_b = FIXED_AUTO_VENDOR_TF32_CELLS[1];
    let hot_d = FIXED_AUTO_VENDOR_TF32_CELLS[3];
    for has_bias in [false, true] {
        for (aligned, known, expected) in [
            (true, true, InferenceTile::Tf32Sm120M128S2),
            (false, true, InferenceTile::Tf32Sm120M64S2),
            (true, false, InferenceTile::Tf32Sm120M64S2),
        ] {
            assert_eq!(
                fixed_auto_vendor_expected_tf32_tile(
                    hot_a,
                    (12, 0),
                    170,
                    (13, 2),
                    has_bias,
                    aligned,
                    known
                ),
                expected
            );
        }
        assert_eq!(
            fixed_auto_vendor_expected_tf32_tile(
                hot_b,
                (12, 0),
                170,
                (13, 2),
                has_bias,
                true,
                true
            ),
            InferenceTile::Tf32Sm120M128S2
        );
        assert_eq!(
            fixed_auto_vendor_expected_tf32_tile(
                hot_d,
                (12, 0),
                170,
                (13, 2),
                has_bias,
                true,
                true
            ),
            InferenceTile::Tf32Sm120M64S2PairStore
        );
        assert_eq!(
            fixed_auto_vendor_expected_tf32_tile(
                hot_d,
                (12, 0),
                170,
                (13, 2),
                has_bias,
                false,
                true
            ),
            InferenceTile::Tf32Sm120M64S2ProducerWarp
        );
        for (device_cc, sm_count, nvrtc_version) in [
            ((12, 0), 169, (13, 2)),
            ((12, 1), 170, (13, 2)),
            ((12, 0), 170, (12, 8)),
            ((12, 0), 170, (13, 0)),
            ((12, 0), 170, (13, 3)),
        ] {
            assert_eq!(
                fixed_auto_vendor_expected_tf32_tile(
                    hot_b,
                    device_cc,
                    sm_count,
                    nvrtc_version,
                    has_bias,
                    true,
                    true
                ),
                InferenceTile::Tf32Sm120M64S2
            );
            assert_eq!(
                fixed_auto_vendor_expected_tf32_tile(
                    hot_d,
                    device_cc,
                    sm_count,
                    nvrtc_version,
                    has_bias,
                    true,
                    true
                ),
                InferenceTile::Tf32Sm120M64S2
            );
        }
    }
}

#[test]
fn fixed_auto_vendor_mixed_expectation_tracks_the_qualified_compiler_cell() {
    let hot_b = FIXED_AUTO_VENDOR_MIXED_CELLS[1];
    assert_eq!(
        fixed_auto_vendor_expected_mixed_tile(hot_b, (12, 0), 170, (13, 2)),
        InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S3)
    );
    for (device_cc, sm_count, nvrtc_version) in [
        ((12, 0), 169, (13, 2)),
        ((12, 1), 170, (13, 2)),
        ((12, 0), 170, (12, 8)),
        ((12, 0), 170, (13, 0)),
    ] {
        assert_eq!(
            fixed_auto_vendor_expected_mixed_tile(hot_b, device_cc, sm_count, nvrtc_version,),
            InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N128Bk32S2)
        );
    }
}

fn configure_fixed_auto_vendor_custom(ctx: &GpuCtx, policy: F32TriadPolicy) {
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_bi_tensor_cores(false);
    ctx.set_f32_triad_policy(policy);
}

fn configure_fixed_auto_vendor_vendor(ctx: &GpuCtx, policy: F32TriadPolicy) {
    ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_bi_tensor_cores(false);
    ctx.set_f32_triad_policy(policy);
}

fn launch_fixed_auto_vendor_custom(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
) -> InferenceTile {
    inference_forward(
        ctx,
        operands.c,
        operands.x,
        operands.w,
        operands.bias_ptr,
        (shape.m, shape.k, shape.n),
    )
    .expect("production Fixed AUTO launch")
}

fn launch_fixed_auto_vendor_vendor(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
) {
    gpu_gemm_typed_forward_raw(
        ctx,
        operands.c,
        operands.x,
        operands.w,
        operands.bias_ptr,
        (shape.m, shape.k, shape.n),
    )
    .expect("fast cuBLAS launch");
}

fn fixed_auto_vendor_custom_window_us(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    policy: F32TriadPolicy,
    iterations: usize,
) -> f64 {
    configure_fixed_auto_vendor_custom(ctx, policy);
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record Fixed AUTO window start");
    for _ in 0..iterations {
        launch_fixed_auto_vendor_custom(ctx, operands, shape);
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record Fixed AUTO window end");
    f64::from(start.elapsed_ms(&end).expect("measure Fixed AUTO window")) * 1000.0
        / iterations as f64
}

fn fixed_auto_vendor_vendor_window_us(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    policy: F32TriadPolicy,
    iterations: usize,
) -> f64 {
    configure_fixed_auto_vendor_vendor(ctx, policy);
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record fast cuBLAS window start");
    for _ in 0..iterations {
        launch_fixed_auto_vendor_vendor(ctx, operands, shape);
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record fast cuBLAS window end");
    f64::from(start.elapsed_ms(&end).expect("measure fast cuBLAS window")) * 1000.0
        / iterations as f64
}

fn fixed_auto_vendor_iterations(pilot_us: f64) -> usize {
    assert!(
        pilot_us.is_finite() && pilot_us > 0.0,
        "pilot latency must be finite and positive, got {pilot_us}"
    );
    (5000.0 / pilot_us).ceil().clamp(1.0, 4096.0) as usize
}

fn run_fixed_auto_vendor_cell(
    ctx: &GpuCtx,
    device_cc: (u32, u32),
    sm_count: u32,
    row: FixedAutoVendorRow,
    cell: FixedAutoVendorCell,
    has_bias: bool,
) {
    let shape = cell.shape;
    let a =
        DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, row.input_dtype).expect("A allocation");
    let b =
        DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, row.input_dtype).expect("B allocation");
    let custom = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, row.output_dtype)
        .expect("custom allocation");
    let vendor = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, row.output_dtype)
        .expect("vendor allocation");
    a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0xa170_0005))
        .expect("A upload");
    b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0xb170_0005))
        .expect("B upload");
    let bias = has_bias.then(|| {
        let bias =
            DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("bias allocation");
        bias.upload_f32(&ctx.stream, &synth(shape.n, 0xb1a5_0005))
            .expect("bias upload");
        bias
    });
    let custom_operands = InferenceFwdOperands {
        c: typed(&custom, row.output_dtype),
        x: typed(&a, row.input_dtype),
        w: typed(&b, row.input_dtype),
        bias_ptr: bias.as_ref().map(DtypedBuf::cached_ptr),
    };
    let vendor_operands = InferenceFwdOperands {
        c: typed(&vendor, row.output_dtype),
        ..custom_operands
    };

    configure_fixed_auto_vendor_custom(ctx, row.policy);
    let selected = launch_fixed_auto_vendor_custom(ctx, custom_operands, shape);
    let expected = if row.name == "f32_exact" {
        fixed_auto_vendor_expected_exact_tile(
            cell,
            device_cc,
            sm_count,
            ctx.kernels.compiler_identity().nvrtc_version,
            ctx.kernels.compiler_identity().nvrtc_library_known,
            has_bias,
            ctx.kernels
                .fixed_sm120_fma_postbias
                .as_ref()
                .is_some_and(|kernels| kernels.m128n64_t256.is_some()),
        )
    } else if row.name == "tf32" {
        fixed_auto_vendor_expected_tf32_tile(
            cell,
            device_cc,
            sm_count,
            ctx.kernels.compiler_identity().nvrtc_version,
            has_bias,
            custom_operands.c.ptr.is_multiple_of(8),
            ctx.kernels.compiler_identity().nvrtc_library_known,
        )
    } else if row.output_dtype == WeightDtype::F32
        && matches!(row.input_dtype, WeightDtype::Bf16 | WeightDtype::F16)
    {
        fixed_auto_vendor_expected_mixed_tile(
            cell,
            device_cc,
            sm_count,
            ctx.kernels.compiler_identity().nvrtc_version,
        )
    } else if row.output_dtype == row.input_dtype
        && matches!(row.input_dtype, WeightDtype::Bf16 | WeightDtype::F16)
    {
        fixed_auto_vendor_expected_half_tile(
            cell,
            row.input_dtype,
            device_cc,
            sm_count,
            ctx.kernels.compiler_identity().nvrtc_version,
            has_bias,
        )
    } else {
        cell.expected
    };
    assert_eq!(
        selected, expected,
        "unexpected production Fixed AUTO tile for row={} cell={} M{} K{} N{}",
        row.name, cell.label, shape.m, shape.k, shape.n
    );
    ctx.stream
        .synchronize()
        .expect("selector assertion synchronization");

    configure_fixed_auto_vendor_custom(ctx, row.policy);
    for _ in 0..128 {
        launch_fixed_auto_vendor_custom(ctx, custom_operands, shape);
    }
    ctx.stream.synchronize().expect("Fixed AUTO warmup sync");
    configure_fixed_auto_vendor_vendor(ctx, row.policy);
    for _ in 0..128 {
        launch_fixed_auto_vendor_vendor(ctx, vendor_operands, shape);
    }
    ctx.stream.synchronize().expect("fast cuBLAS warmup sync");

    let custom_pilot_us =
        fixed_auto_vendor_custom_window_us(ctx, custom_operands, shape, row.policy, 16);
    let vendor_pilot_us =
        fixed_auto_vendor_vendor_window_us(ctx, vendor_operands, shape, row.policy, 16);
    let custom_iterations = fixed_auto_vendor_iterations(custom_pilot_us);
    let vendor_iterations = fixed_auto_vendor_iterations(vendor_pilot_us);

    for custom_first in [true, false] {
        let mut custom_us = Vec::with_capacity(101);
        let mut vendor_us = Vec::with_capacity(101);
        let mut ratios = Vec::with_capacity(101);
        for _ in 0..101 {
            let (custom_elapsed_us, vendor_elapsed_us) = if custom_first {
                (
                    fixed_auto_vendor_custom_window_us(
                        ctx,
                        custom_operands,
                        shape,
                        row.policy,
                        custom_iterations,
                    ),
                    fixed_auto_vendor_vendor_window_us(
                        ctx,
                        vendor_operands,
                        shape,
                        row.policy,
                        vendor_iterations,
                    ),
                )
            } else {
                let vendor_elapsed_us = fixed_auto_vendor_vendor_window_us(
                    ctx,
                    vendor_operands,
                    shape,
                    row.policy,
                    vendor_iterations,
                );
                let custom_elapsed_us = fixed_auto_vendor_custom_window_us(
                    ctx,
                    custom_operands,
                    shape,
                    row.policy,
                    custom_iterations,
                );
                (custom_elapsed_us, vendor_elapsed_us)
            };
            assert!(
                custom_elapsed_us.is_finite()
                    && custom_elapsed_us > 0.0
                    && vendor_elapsed_us.is_finite()
                    && vendor_elapsed_us > 0.0,
                "measured latencies must be finite and positive"
            );
            custom_us.push(custom_elapsed_us);
            vendor_us.push(vendor_elapsed_us);
            ratios.push(custom_elapsed_us / vendor_elapsed_us);
        }
        custom_us.sort_by(f64::total_cmp);
        vendor_us.sort_by(f64::total_cmp);
        ratios.sort_by(f64::total_cmp);
        let comparator_order = if custom_first {
            "custom_then_vendor"
        } else {
            "vendor_then_custom"
        };
        println!(
            concat!(
                "{{\"schema\":\"MambaBiFixedAutoVendorV2\",",
                "\"suite\":\"fixed_production_auto_vs_fast_cublas_hot_and_selector_census\",",
                "\"tuning_table_revision\":{},\"device_cc\":\"{}.{}\",\"sm_count\":{},",
                "\"row\":\"{}\",\"input_dtype\":\"{}\",\"output_dtype\":\"{}\",\"f32_policy\":\"{}\",",
                "\"op\":\"nn\",\"bias\":{},\"cell_label\":\"{}\",\"m\":{},\"k\":{},\"n\":{},",
                "\"auto_tile\":\"{:?}\",\"cublas_mode\":\"fast_tf32_allowed\",",
                "\"comparator_order\":\"{}\",\"warmups\":128,\"pilot_iterations\":16,",
                "\"target_window_ms\":5.0,\"windows\":101,",
                "\"custom_iterations\":{},\"vendor_iterations\":{},",
                "\"custom_p50_us\":{:.9},\"custom_p95_us\":{:.9},",
                "\"vendor_p50_us\":{:.9},\"vendor_p95_us\":{:.9},",
                "\"ratio_p50\":{:.9},\"ratio_p95\":{:.9}}}"
            ),
            TUNING_TABLE_REVISION,
            device_cc.0,
            device_cc.1,
            sm_count,
            row.name,
            row.input_dtype.as_str(),
            row.output_dtype.as_str(),
            row.policy_name,
            has_bias,
            cell.label,
            shape.m,
            shape.k,
            shape.n,
            selected,
            comparator_order,
            custom_iterations,
            vendor_iterations,
            percentile(&custom_us, 0.50),
            percentile(&custom_us, 0.95),
            percentile(&vendor_us, 0.50),
            percentile(&vendor_us, 0.95),
            percentile(&ratios, 0.50),
            percentile(&ratios, 0.95),
        );
    }
}

#[test]
#[ignore = "requires a quiet 170-SM CC12 CUDA device and emits production AUTO/vendor evidence"]
fn fixed_production_auto_vs_fast_cublas_hot_and_selector_census() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let device_cc = device.compute_capability;
    let sm_count = device.multiprocessor_count();
    if !matches!(device_cc, (12, 0) | (12, 1)) || sm_count != 170 {
        eprintln!(
            "skipping the production Inference AUTO/vendor survey on CC{}.{} with {} SMs; CC12.0/12.1 with exactly 170 SMs required",
            device_cc.0, device_cc.1, sm_count
        );
        return;
    }
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert!(ctx.tf32(), "fast cuBLAS TF32 must remain enabled");
    println!();
    let rows = [
        FixedAutoVendorRow {
            name: "bf16",
            input_dtype: WeightDtype::Bf16,
            output_dtype: WeightDtype::Bf16,
            policy: F32TriadPolicy::ExactScalarFma,
            policy_name: "not_applicable",
            cells: FIXED_AUTO_VENDOR_HALF_CELLS,
        },
        FixedAutoVendorRow {
            name: "f16",
            input_dtype: WeightDtype::F16,
            output_dtype: WeightDtype::F16,
            policy: F32TriadPolicy::ExactScalarFma,
            policy_name: "not_applicable",
            cells: FIXED_AUTO_VENDOR_HALF_CELLS,
        },
        FixedAutoVendorRow {
            name: "bf16_f32",
            input_dtype: WeightDtype::Bf16,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::ExactScalarFma,
            policy_name: "not_applicable",
            cells: FIXED_AUTO_VENDOR_MIXED_CELLS,
        },
        FixedAutoVendorRow {
            name: "f16_f32",
            input_dtype: WeightDtype::F16,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::ExactScalarFma,
            policy_name: "not_applicable",
            cells: FIXED_AUTO_VENDOR_MIXED_CELLS,
        },
        FixedAutoVendorRow {
            name: "tf32",
            input_dtype: WeightDtype::F32,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::AllowDeterministicTf32,
            policy_name: "allow_deterministic_tf32",
            cells: FIXED_AUTO_VENDOR_TF32_CELLS,
        },
        FixedAutoVendorRow {
            name: "f32_exact",
            input_dtype: WeightDtype::F32,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::ExactScalarFma,
            policy_name: "exact_scalar_fma",
            cells: FIXED_AUTO_VENDOR_EXACT_CELLS,
        },
    ];
    let row_filter = std::env::var("MAMBA_FIXED_AUTO_VENDOR_ROW").ok();
    if let Some(filter) = row_filter.as_deref() {
        assert!(
            rows.iter().any(|row| row.name == filter),
            "unknown Fixed AUTO/vendor row filter: {filter}"
        );
    }
    let cell_filter = std::env::var("MAMBA_FIXED_AUTO_VENDOR_CELL").ok();
    if let Some(filter) = cell_filter.as_deref() {
        assert!(
            rows.iter()
                .flat_map(|row| row.cells)
                .any(|cell| cell.label == filter),
            "unknown Fixed AUTO/vendor cell filter: {filter}"
        );
    }
    let has_bias = match std::env::var("MAMBA_FIXED_AUTO_VENDOR_BIAS").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(_) => false,
        Ok(value) => panic!("MAMBA_FIXED_AUTO_VENDOR_BIAS must be 0 or 1, got {value}"),
    };
    for row in rows {
        if row_filter
            .as_deref()
            .is_some_and(|filter| filter != row.name)
        {
            continue;
        }
        for &cell in row.cells {
            if cell_filter
                .as_deref()
                .is_some_and(|filter| filter != cell.label)
            {
                continue;
            }
            run_fixed_auto_vendor_cell(&ctx, device_cc, sm_count, row, cell, has_bias);
        }
    }
}

// This comparator deliberately bypasses effective_compute: the legacy Fixed
// smoke tests use COMPUTE_32F for both F32 modes, which is neither an explicit
// FAST_TF32 denominator nor a PEDANTIC exact-F32 denominator.
fn fixed_ada_vendor_launch(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    compute: cudarc::cublas::sys::cublasComputeType_t,
) {
    use std::ffi::{c_int, c_void};

    let beta = if let Some(bias_ptr) = operands.bias_ptr {
        // Match the production vendor epilogue, including its timed broadcast.
        // A half output rounds the broadcast bias to its storage dtype before
        // GEMM; the independent reference below keeps its output in F32.
        let kernel = match operands.c.dtype {
            WeightDtype::F32 => &ctx.kernels.bias_broadcast,
            dtype => ctx.kernels.bias_broadcast_typed.get(dtype),
        };
        let rows = shape.m as c_int;
        let cols = shape.n as c_int;
        let mut launch = ctx.stream.launch_builder(kernel);
        launch.arg(&operands.c.ptr);
        launch.arg(&bias_ptr);
        launch.arg(&rows);
        launch.arg(&cols);
        unsafe { launch.launch(mamba_rs::mamba_ssm::gpu::launch::grid_1d(shape.m * shape.n)) }
            .expect("Ada vendor bias broadcast");
        1.0f32
    } else {
        0.0f32
    };
    let alpha = 1.0f32;
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *ctx.blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            shape.n as c_int,
            shape.m as c_int,
            shape.k as c_int,
            &alpha as *const f32 as *const c_void,
            operands.w.ptr as *const c_void,
            operands.w.dtype.cuda_data_type(),
            shape.n as c_int,
            operands.x.ptr as *const c_void,
            operands.x.dtype.cuda_data_type(),
            shape.k as c_int,
            &beta as *const f32 as *const c_void,
            operands.c.ptr as *mut c_void,
            operands.c.dtype.cuda_data_type(),
            shape.n as c_int,
            compute,
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .expect("Ada explicit-compute vendor GEMM");
    }
}

fn fixed_ada_event_window_us(ctx: &GpuCtx, iterations: usize, mut launch: impl FnMut()) -> f64 {
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("Ada paired window start");
    for _ in 0..iterations {
        launch();
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("Ada paired window end");
    let elapsed = f64::from(start.elapsed_ms(&end).expect("Ada paired event timing")) * 1000.0
        / iterations as f64;
    assert!(
        elapsed.is_finite() && elapsed > 0.0,
        "invalid Ada timing: {elapsed}"
    );
    elapsed
}

fn fixed_sm120_fma_spike_tile(value: Option<&str>) -> Result<InferenceTile, String> {
    Ok(match value {
        Some("m64n128") => InferenceTile::F32Sm120TmaFmaM64N128,
        Some("m128n64") | None => InferenceTile::F32Sm120TmaFmaM128N64,
        Some("postbias_m64n128") => InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128,
        Some("postbias_m128n64") => InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
        Some("postbias_m128n64_k4") => InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64K4,
        Some("postbias_m128n64_t256") => InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256,
        Some("postbias_m128n96") => InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96,
        Some("nobias_m128n64_t256") => InferenceTile::F32Sm120TmaFmaFixedNoBiasM128N64T256,
        Some("copyplan") => InferenceTile::F32Sm120N64CopyPlan,
        Some("copyplan_t256") => InferenceTile::F32Sm120N64CopyPlanT256,
        Some("copyplan_m128n64_t256") => InferenceTile::F32Sm120M128N64CopyPlanT256,
        Some(value) => {
            return Err(format!(
                "MAMBA_FIXED_SM120_FMA_SPIKE_TILE must be m128n64, m64n128, postbias_m128n64, postbias_m128n64_k4, postbias_m128n64_t256, postbias_m64n128, postbias_m128n96, nobias_m128n64_t256, copyplan, copyplan_t256 or copyplan_m128n64_t256, got {value}"
            ));
        }
    })
}

#[test]
#[ignore = "requires a quiet RTX5090 and screens an exact SM120 TMA-FMA route on a Fixed hot cell"]
fn fixed_sm120_exact_tma_fma_b0_spike() {
    use cudarc::cublas::sys::cublasComputeType_t;

    assert_eq!(
        std::env::var("MAMBA_FIXED_SM120_FMA_SPIKE").as_deref(),
        Ok("1"),
        "set MAMBA_FIXED_SM120_FMA_SPIKE=1 for the explicit hot-cell spike"
    );
    assert_ne!(
        std::env::var("NVIDIA_TF32_OVERRIDE").as_deref(),
        Ok("0"),
        "NVIDIA_TF32_OVERRIDE=0 disables the explicit FAST_TF32 denominator"
    );
    if cfg!(debug_assertions) {
        panic!("hot-cell spike requires --release");
    }
    fixed_sm120_tf32_bd_environment_preflight("Fixed exact-TMA hot-cell spike")
        .expect("quiet-GPU preflight");
    let device = GpuDevice::new(0).expect("hot-cell spike CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let ctx = GpuCtx::new(&device).expect("hot-cell spike GPU context");
    assert_eq!(ctx.kernels.compiler_identity().nvrtc_version, (13, 2));
    assert!(ctx.kernels.compiler_identity().nvrtc_library_known);
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFma);

    let cell = std::env::var("MAMBA_FIXED_SM120_FMA_SPIKE_CELL").unwrap_or_else(|_| "b".into());
    let dims = match cell.as_str() {
        "a" => (4_621usize, 384usize, 1_928usize),
        "b" => (4_621, 768, 2_304),
        "c" => (4_621, 1_928, 384),
        "d" => (2_048, 768, 2_304),
        "e" => (2_048, 2_304, 768),
        other => panic!("unknown exact-TMA Fixed hot cell: {other}"),
    };
    let has_bias = match std::env::var("MAMBA_FIXED_SM120_FMA_SPIKE_BIAS").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(_) => false,
        Ok(value) => panic!("MAMBA_FIXED_SM120_FMA_SPIKE_BIAS must be 0 or 1, got {value}"),
    };
    let candidate_tile = fixed_sm120_fma_spike_tile(
        std::env::var("MAMBA_FIXED_SM120_FMA_SPIKE_TILE")
            .ok()
            .as_deref(),
    )
    .expect("exact-TMA spike tile selection");
    let expected_candidate_symbol = match candidate_tile {
        InferenceTile::F32Sm120N64CopyPlan => "nn_sm120_f32_n64_copyplan",
        InferenceTile::F32Sm120M128N64CopyPlanT256 => "nn_sm120_f32_n64_copyplan_m128n64_t256",
        InferenceTile::F32Sm120N64CopyPlanT256 => "nn_sm120_f32_n64_copyplan_t256",
        InferenceTile::F32Sm120TmaFmaFixedNoBiasM128N64T256 => {
            "nn_sm120_tma_fma_nobias_m128n64_t256_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256 => {
            "nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64K4 => {
            "nn_sm120_tma_fma_postbias_m128n64_bk16_s2_k4"
        }
        InferenceTile::F32Sm120TmaFmaM128N64 => "nn_sm120_tma_fma_m128n64_bk16_s2",
        InferenceTile::F32Sm120TmaFmaM64N128 => "nn_sm120_tma_fma_m64n128_bk16_s2",
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64 => {
            "nn_sm120_tma_fma_postbias_m128n64_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128 => {
            "nn_sm120_tma_fma_postbias_m64n128_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96 => {
            "nn_sm120_tma_fma_postbias_m128n96_bk16_s2"
        }
        _ => unreachable!(),
    };
    let shape = InferenceShape {
        m: dims.0,
        k: dims.1,
        n: dims.2,
    };
    let mut a = GpuBuffer::from_cpu(&ctx.stream, &synth(shape.m * shape.k, 0xf120_a001))
        .expect("hot-cell spike A");
    let mut b = GpuBuffer::from_cpu(&ctx.stream, &synth(shape.k * shape.n, 0xf120_b001))
        .expect("hot-cell spike B");
    let mut bias = has_bias.then(|| {
        GpuBuffer::from_cpu(&ctx.stream, &synth(shape.n, 0xf120_b1a5)).expect("hot-cell spike bias")
    });
    let candidate = GpuBuffer::zeros(&ctx.stream, shape.m * shape.n).expect("candidate C");
    let auto = GpuBuffer::zeros(&ctx.stream, shape.m * shape.n).expect("AUTO C");
    let legacy = GpuBuffer::zeros(&ctx.stream, shape.m * shape.n).expect("Legacy C");
    let mut vendor = GpuBuffer::zeros(&ctx.stream, shape.m * shape.n).expect("vendor C");
    let mut vendor_fast = GpuBuffer::zeros(&ctx.stream, shape.m * shape.n).expect("FAST vendor C");
    let candidate_operands = InferenceFwdOperands {
        c: TypedPtr {
            ptr: candidate.cached_ptr(),
            dtype: WeightDtype::F32,
        },
        x: TypedPtr {
            ptr: a.cached_ptr(),
            dtype: WeightDtype::F32,
        },
        w: TypedPtr {
            ptr: b.cached_ptr(),
            dtype: WeightDtype::F32,
        },
        bias_ptr: bias.as_ref().map(GpuBuffer::cached_ptr),
    };
    let auto_operands = InferenceFwdOperands {
        c: TypedPtr {
            ptr: auto.cached_ptr(),
            dtype: WeightDtype::F32,
        },
        ..candidate_operands
    };
    let legacy_operands = InferenceFwdOperands {
        c: TypedPtr {
            ptr: legacy.cached_ptr(),
            dtype: WeightDtype::F32,
        },
        ..candidate_operands
    };
    let vendor_operands = InferenceFwdOperands {
        c: TypedPtr {
            ptr: vendor.cached_ptr(),
            dtype: WeightDtype::F32,
        },
        ..candidate_operands
    };
    let vendor_fast_operands = InferenceFwdOperands {
        c: TypedPtr {
            ptr: vendor_fast.cached_ptr(),
            dtype: WeightDtype::F32,
        },
        ..candidate_operands
    };

    let launch_candidate = || {
        inference_forward_with_tile(&ctx, candidate_operands, shape, candidate_tile)
            .expect("Fixed exact SM120 TMA-FMA hot-cell forced launch");
    };
    let launch_auto = || {
        mamba_rs::mamba_ssm::gpu::gemm_bi_inference::inference_forward(
            &ctx,
            auto_operands.c,
            auto_operands.x,
            auto_operands.w,
            auto_operands.bias_ptr,
            dims,
        )
        .expect("Fixed exact SM120 hot-cell AUTO launch")
    };
    let expected_auto = match (cell.as_str(), has_bias) {
        ("a", false) => InferenceTile::F32Sm120TmaFmaM64N128,
        ("a", true)
            if ctx
                .kernels
                .fixed_sm120_fma_postbias
                .as_ref()
                .is_some_and(|kernels| kernels.m128n64_t256.is_some()) =>
        {
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
        }
        ("a", true) => InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
        ("b", false) => InferenceTile::F32Sm120TmaFmaM128N64,
        ("b", true) => InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
        ("c", true) => InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96,
        ("d" | "e", _) => InferenceTile::F32Sm120N64CopyPlan,
        ("c", false) => InferenceTile::Legacy,
        _ => unreachable!(),
    };
    let launch_auto_checked = || {
        assert_eq!(launch_auto(), expected_auto);
    };
    let launch_legacy = || {
        inference_forward_with_tile(&ctx, legacy_operands, shape, InferenceTile::Legacy)
            .expect("hot-cell Legacy launch");
    };
    let launch_vendor = || {
        fixed_ada_vendor_launch(
            &ctx,
            vendor_operands,
            shape,
            cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
        );
    };
    let launch_vendor_fast = || {
        fixed_ada_vendor_launch(
            &ctx,
            vendor_fast_operands,
            shape,
            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
        );
    };

    launch_candidate();
    launch_auto_checked();
    launch_legacy();
    launch_vendor();
    launch_vendor_fast();
    ctx.stream
        .synchronize()
        .expect("hot-cell initial synchronization");
    let candidate_bits = candidate
        .to_cpu(&ctx.stream)
        .expect("candidate download")
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let legacy_bits = legacy
        .to_cpu(&ctx.stream)
        .expect("Legacy download")
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let auto_bits = auto
        .to_cpu(&ctx.stream)
        .expect("AUTO download")
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let vendor_bits = vendor
        .to_cpu(&ctx.stream)
        .expect("PEDANTIC vendor download")
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let vendor_fast_bits = vendor_fast
        .to_cpu(&ctx.stream)
        .expect("FAST vendor download")
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(
        candidate_bits, legacy_bits,
        "TMA-FMA must retain Fixed hot-cell bits"
    );
    assert_eq!(
        auto_bits, legacy_bits,
        "AUTO must retain Fixed hot-cell bits"
    );

    for _ in 0..128 {
        launch_candidate();
        launch_auto_checked();
        launch_legacy();
        launch_vendor();
        launch_vendor_fast();
    }
    ctx.stream
        .synchronize()
        .expect("hot-cell eager warmup synchronization");
    let candidate_graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            launch_candidate();
            Ok(())
        })
    }
    .expect("capture exact SM120 TMA-FMA hot-cell graph");
    let auto_graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            launch_auto_checked();
            Ok(())
        })
    }
    .expect("capture hot-cell AUTO graph");
    let legacy_graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            launch_legacy();
            Ok(())
        })
    }
    .expect("capture hot-cell Legacy graph");
    let vendor_graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            launch_vendor();
            Ok(())
        })
    }
    .expect("capture PEDANTIC hot-cell graph");
    let vendor_fast_graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            launch_vendor_fast();
            Ok(())
        })
    }
    .expect("capture FAST_TF32 hot-cell graph");
    let _ = fixed_explicit_vendor_graph_inventory(
        &vendor_graph,
        "PEDANTIC hot-cell vendor",
        None,
        None,
        has_bias.then_some("bias_broadcast"),
    );
    let _ = fixed_explicit_vendor_graph_inventory(
        &vendor_fast_graph,
        "FAST_TF32 hot-cell vendor",
        None,
        None,
        has_bias.then_some("bias_broadcast"),
    );
    let candidate_symbol = single_graph_kernel_name(&candidate_graph, "TMA-FMA hot cell");
    assert_eq!(candidate_symbol, expected_candidate_symbol);
    assert_sm120_exact_tma_graph_contract(
        &candidate_graph,
        candidate.cached_ptr(),
        shape,
        candidate_operands.bias_ptr,
        candidate_tile,
    );
    let expected_auto_symbol = match expected_auto {
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256 => {
            "nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaM128N64 => "nn_sm120_tma_fma_m128n64_bk16_s2",
        InferenceTile::F32Sm120TmaFmaM64N128 => "nn_sm120_tma_fma_m64n128_bk16_s2",
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64 => {
            "nn_sm120_tma_fma_postbias_m128n64_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128 => {
            "nn_sm120_tma_fma_postbias_m64n128_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96 => {
            "nn_sm120_tma_fma_postbias_m128n96_bk16_s2"
        }
        InferenceTile::F32Sm120N64CopyPlan => "nn_sm120_f32_n64_copyplan",
        InferenceTile::F32N128S2 => "f32_f32_n128_s2",
        InferenceTile::Legacy => "f32_f32_s2",
        _ => unreachable!(),
    };
    assert_eq!(
        single_graph_kernel_name(&auto_graph, "hot-cell AUTO"),
        expected_auto_symbol
    );
    if matches!(
        expected_auto,
        InferenceTile::F32Sm120TmaFmaM128N64
            | InferenceTile::F32Sm120TmaFmaM64N128
            | InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
            | InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64
            | InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128
            | InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96
    ) {
        assert_sm120_exact_tma_graph_contract(
            &auto_graph,
            auto.cached_ptr(),
            shape,
            auto_operands.bias_ptr,
            expected_auto,
        );
    }
    assert_eq!(
        single_graph_kernel_name(&legacy_graph, "hot-cell Legacy"),
        "f32_f32_s2"
    );
    vendor
        .zero(&ctx.stream)
        .expect("poison PEDANTIC vendor output before graph replay");
    vendor_fast
        .zero(&ctx.stream)
        .expect("poison FAST vendor output before graph replay");
    vendor_graph
        .launch()
        .expect("PEDANTIC vendor poison replay");
    vendor_fast_graph
        .launch()
        .expect("FAST vendor poison replay");
    ctx.stream
        .synchronize()
        .expect("vendor poison replay synchronization");
    assert_eq!(
        vendor
            .to_cpu(&ctx.stream)
            .expect("PEDANTIC poison replay download")
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        vendor_bits,
        "PEDANTIC vendor graph did not restore eager output after poison"
    );
    assert_eq!(
        vendor_fast
            .to_cpu(&ctx.stream)
            .expect("FAST poison replay download")
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        vendor_fast_bits,
        "FAST vendor graph did not restore eager output after poison"
    );
    for _ in 0..128 {
        candidate_graph.launch().expect("candidate graph warmup");
        auto_graph.launch().expect("AUTO graph warmup");
        legacy_graph.launch().expect("Legacy graph warmup");
        vendor_graph.launch().expect("vendor graph warmup");
        vendor_fast_graph
            .launch()
            .expect("FAST vendor graph warmup");
    }
    ctx.stream
        .synchronize()
        .expect("hot-cell graph warmup synchronization");
    let graph_bits = candidate
        .to_cpu(&ctx.stream)
        .expect("candidate graph download")
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let auto_graph_bits = auto
        .to_cpu(&ctx.stream)
        .expect("AUTO graph download")
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let vendor_graph_bits = vendor
        .to_cpu(&ctx.stream)
        .expect("PEDANTIC vendor graph download")
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let vendor_fast_graph_bits = vendor_fast
        .to_cpu(&ctx.stream)
        .expect("FAST vendor graph download")
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(
        graph_bits, candidate_bits,
        "Fixed exact-TMA graph replay changed hot-cell bits"
    );
    assert_eq!(
        auto_graph_bits, legacy_bits,
        "Fixed AUTO graph replay changed hot-cell bits"
    );
    assert_eq!(
        vendor_graph_bits, vendor_bits,
        "PEDANTIC vendor graph replay changed hot-cell bits"
    );
    assert_eq!(
        vendor_fast_graph_bits, vendor_fast_bits,
        "FAST_TF32 vendor graph replay changed hot-cell bits"
    );

    let windows = std::env::var("MAMBA_FIXED_SM120_FMA_SPIKE_WINDOWS")
        .map_or(21, |value| value.parse::<usize>().expect("spike windows"));
    assert!((1..=101).contains(&windows));
    for path in ["eager", "graph"] {
        for (comparator_name, comparator_graph) in [
            ("current_auto", &auto_graph),
            ("legacy", &legacy_graph),
            ("cublas_pedantic", &vendor_graph),
            ("cublas_fast_tf32", &vendor_fast_graph),
        ] {
            for candidate_outside in [true, false] {
                let candidate_pilot = fixed_ada_event_window_us(&ctx, 16, || {
                    if path == "graph" {
                        candidate_graph.launch().expect("candidate graph pilot");
                    } else {
                        launch_candidate();
                    }
                });
                let comparator_pilot = fixed_ada_event_window_us(&ctx, 16, || {
                    if path == "graph" {
                        comparator_graph.launch().expect("comparator graph pilot");
                    } else {
                        match comparator_name {
                            "current_auto" => launch_auto_checked(),
                            "legacy" => launch_legacy(),
                            "cublas_pedantic" => launch_vendor(),
                            "cublas_fast_tf32" => launch_vendor_fast(),
                            _ => unreachable!(),
                        }
                    }
                });
                let candidate_iterations = fixed_auto_vendor_iterations(candidate_pilot);
                let comparator_iterations = fixed_auto_vendor_iterations(comparator_pilot);
                let mut ratios = Vec::with_capacity(windows);
                let mut candidate_us = Vec::with_capacity(windows);
                let mut comparator_us = Vec::with_capacity(windows);
                for _ in 0..windows {
                    let time_candidate = || {
                        fixed_ada_event_window_us(&ctx, candidate_iterations, || {
                            if path == "graph" {
                                candidate_graph.launch().expect("candidate graph timing");
                            } else {
                                launch_candidate();
                            }
                        })
                    };
                    let time_comparator = || {
                        fixed_ada_event_window_us(&ctx, comparator_iterations, || {
                            if path == "graph" {
                                comparator_graph.launch().expect("comparator graph timing");
                            } else {
                                match comparator_name {
                                    "current_auto" => launch_auto_checked(),
                                    "legacy" => launch_legacy(),
                                    "cublas_pedantic" => launch_vendor(),
                                    "cublas_fast_tf32" => launch_vendor_fast(),
                                    _ => unreachable!(),
                                }
                            }
                        })
                    };
                    let (candidate_elapsed, comparator_elapsed) = if candidate_outside {
                        let candidate_first = time_candidate();
                        let comparator_first = time_comparator();
                        let comparator_second = time_comparator();
                        let candidate_second = time_candidate();
                        (
                            (candidate_first + candidate_second) * 0.5,
                            (comparator_first + comparator_second) * 0.5,
                        )
                    } else {
                        let comparator_first = time_comparator();
                        let candidate_first = time_candidate();
                        let candidate_second = time_candidate();
                        let comparator_second = time_comparator();
                        (
                            (candidate_first + candidate_second) * 0.5,
                            (comparator_first + comparator_second) * 0.5,
                        )
                    };
                    candidate_us.push(candidate_elapsed);
                    comparator_us.push(comparator_elapsed);
                    ratios.push(candidate_elapsed / comparator_elapsed);
                }
                candidate_us.sort_by(f64::total_cmp);
                comparator_us.sort_by(f64::total_cmp);
                ratios.sort_by(f64::total_cmp);
                println!(
                    concat!(
                        "{{\"schema\":\"MambaBiFixedSm120FmaHotCellSpikeV2\",",
                        "\"cell\":\"{}\",\"bias\":{},\"dims_mkn\":[{},{},{}],",
                        "\"path\":\"{}\",\"comparator\":\"{}\",\"order\":\"{}\",",
                        "\"windows\":{},\"candidate_symbol\":\"{}\",",
                        "\"candidate_p50_us\":{:.9},\"candidate_p95_us\":{:.9},",
                        "\"comparator_p50_us\":{:.9},\"comparator_p95_us\":{:.9},",
                        "\"ratio_p50\":{:.9},\"ratio_p95\":{:.9}}}"
                    ),
                    cell,
                    has_bias,
                    shape.m,
                    shape.k,
                    shape.n,
                    path,
                    comparator_name,
                    if candidate_outside { "abba" } else { "baab" },
                    windows,
                    candidate_symbol,
                    percentile(&candidate_us, 0.50),
                    percentile(&candidate_us, 0.95),
                    percentile(&comparator_us, 0.50),
                    percentile(&comparator_us, 0.95),
                    percentile(&ratios, 0.50),
                    percentile(&ratios, 0.95),
                );
            }
        }
    }

    // Keep the managed input allocations live through every graph replay.
    a.zero(&ctx.stream).expect("keep A live");
    b.zero(&ctx.stream).expect("keep B live");
    if let Some(bias) = bias.as_mut() {
        bias.zero(&ctx.stream).expect("keep bias live");
    }
}

#[test]
#[ignore = "requires a quiet RTX5090 and balances all exact TMA-FMA B0 tile orders"]
fn fixed_sm120_exact_tma_fma_b0_tile_tournament() {
    assert_eq!(
        std::env::var("MAMBA_FIXED_SM120_FMA_TOURNAMENT").as_deref(),
        Ok("1"),
        "set MAMBA_FIXED_SM120_FMA_TOURNAMENT=1 for the explicit B0 tournament"
    );
    if cfg!(debug_assertions) {
        panic!("B0 tournament requires --release");
    }
    let device = GpuDevice::new(0).expect("B0 tournament CUDA device");
    assert_eq!(device.compute_capability, (12, 0));
    assert_eq!(device.multiprocessor_count(), 170);
    let contexts = [
        GpuCtx::new(&device).expect("M128N64 context"),
        GpuCtx::new(&device).expect("M64N128 context"),
        GpuCtx::new(&device).expect("M64N64 context"),
    ];
    let symbols = [
        "nn_sm120_tma_fma_m128n64_bk16_s2",
        "nn_sm120_tma_fma_m64n128_bk16_s2",
        "nn_sm120_tma_fma_m64n64_bk16_s2",
    ];
    let requests = symbols.map(|symbol| {
        let route = tf32_route_specs(ModuleKind::TriadSm120)
            .iter()
            .find(|spec| spec.symbol == symbol)
            .unwrap_or_else(|| panic!("missing exact TMA-FMA route {symbol}"))
            .route;
        assert!(route.is_exact_fma());
        PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nn,
            (4_621, 768, 2_304),
            PhysicalQualificationRoute::Tf32Forced(route),
        )
    });
    let mut h0 = qualify_physical_launch(&contexts[0], requests[0]).expect("qualify M128N64");
    let mut h1 = qualify_physical_launch(&contexts[1], requests[1]).expect("qualify M64N128");
    let mut h2 = qualify_physical_launch(&contexts[2], requests[2]).expect("qualify M64N64");
    for (holder, ctx) in [
        (&mut h0, &contexts[0]),
        (&mut h1, &contexts[1]),
        (&mut h2, &contexts[2]),
    ] {
        holder
            .seed_f32_operands(ctx, 0xf120_7001)
            .expect("seed B0 tournament operands");
        holder
            .measure_prevalidated_forced_eager_window_ms(ctx, 1)
            .expect("B0 eager bit launch");
    }
    let eager_bits = [
        h0.f32_output_bits(&contexts[0])
            .expect("M128N64 eager bits"),
        h1.f32_output_bits(&contexts[1])
            .expect("M64N128 eager bits"),
        h2.f32_output_bits(&contexts[2]).expect("M64N64 eager bits"),
    ];
    assert_eq!(
        eager_bits[0], eager_bits[1],
        "M64N128 changed exact B0 bits"
    );
    assert_eq!(eager_bits[0], eager_bits[2], "M64N64 changed exact B0 bits");
    for (holder, ctx) in [
        (&h0, &contexts[0]),
        (&h1, &contexts[1]),
        (&h2, &contexts[2]),
    ] {
        holder
            .measure_graph_window_ms(ctx, 1)
            .expect("B0 graph bit launch");
    }
    assert_eq!(
        eager_bits[0],
        h0.f32_output_bits(&contexts[0])
            .expect("M128N64 graph bits")
    );
    assert_eq!(
        eager_bits[0],
        h1.f32_output_bits(&contexts[1])
            .expect("M64N128 graph bits")
    );
    assert_eq!(
        eager_bits[0],
        h2.f32_output_bits(&contexts[2]).expect("M64N64 graph bits")
    );
    for (holder, symbol) in [(&h0, symbols[0]), (&h1, symbols[1]), (&h2, symbols[2])] {
        assert_eq!(holder.evidence().single_launch_symbol(), Some(symbol));
        assert_eq!(holder.evidence().launch_count(), 1);
    }

    let eager_pilots = [
        h0.measure_prevalidated_forced_eager_window_ms(&contexts[0], 16)
            .expect("M128N64 eager pilot")
            * 1_000.0
            / 16.0,
        h1.measure_prevalidated_forced_eager_window_ms(&contexts[1], 16)
            .expect("M64N128 eager pilot")
            * 1_000.0
            / 16.0,
        h2.measure_prevalidated_forced_eager_window_ms(&contexts[2], 16)
            .expect("M64N64 eager pilot")
            * 1_000.0
            / 16.0,
    ];
    let graph_pilots = [
        h0.measure_graph_window_ms(&contexts[0], 16)
            .expect("M128N64 graph pilot")
            * 1_000.0
            / 16.0,
        h1.measure_graph_window_ms(&contexts[1], 16)
            .expect("M64N128 graph pilot")
            * 1_000.0
            / 16.0,
        h2.measure_graph_window_ms(&contexts[2], 16)
            .expect("M64N64 graph pilot")
            * 1_000.0
            / 16.0,
    ];
    let eager_iterations = eager_pilots.map(fixed_auto_vendor_iterations);
    let graph_iterations = graph_pilots.map(fixed_auto_vendor_iterations);
    let windows = std::env::var("MAMBA_FIXED_SM120_FMA_TOURNAMENT_WINDOWS").map_or(21, |value| {
        value.parse::<usize>().expect("tournament windows")
    });
    assert!((1..=101).contains(&windows));
    let permutations = [
        [0usize, 1usize, 2usize],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for path in ["eager", "graph"] {
        let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| {
            Vec::with_capacity(windows.checked_mul(permutations.len()).unwrap())
        });
        for _ in 0..windows {
            for order in permutations {
                for index in order {
                    let elapsed_ms = match (path, index) {
                        ("eager", 0) => h0
                            .measure_prevalidated_forced_eager_window_ms(
                                &contexts[0],
                                eager_iterations[0],
                            )
                            .expect("M128N64 eager window"),
                        ("eager", 1) => h1
                            .measure_prevalidated_forced_eager_window_ms(
                                &contexts[1],
                                eager_iterations[1],
                            )
                            .expect("M64N128 eager window"),
                        ("eager", 2) => h2
                            .measure_prevalidated_forced_eager_window_ms(
                                &contexts[2],
                                eager_iterations[2],
                            )
                            .expect("M64N64 eager window"),
                        ("graph", 0) => h0
                            .measure_graph_window_ms(&contexts[0], graph_iterations[0])
                            .expect("M128N64 graph window"),
                        ("graph", 1) => h1
                            .measure_graph_window_ms(&contexts[1], graph_iterations[1])
                            .expect("M64N128 graph window"),
                        ("graph", 2) => h2
                            .measure_graph_window_ms(&contexts[2], graph_iterations[2])
                            .expect("M64N64 graph window"),
                        _ => unreachable!(),
                    };
                    let iterations = if path == "eager" {
                        eager_iterations[index]
                    } else {
                        graph_iterations[index]
                    };
                    samples[index].push(elapsed_ms * 1_000.0 / iterations as f64);
                }
            }
        }
        for (index, sample) in samples.iter_mut().enumerate() {
            sample.sort_by(f64::total_cmp);
            println!(
                concat!(
                    "{{\"schema\":\"MambaBiFixedSm120FmaB0TileTournamentV1\",",
                    "\"path\":\"{}\",\"symbol\":\"{}\",\"windows_per_permutation\":{},",
                    "\"balanced_samples\":{},\"iterations\":{},",
                    "\"p50_us\":{:.9},\"p95_us\":{:.9}}}"
                ),
                path,
                symbols[index],
                windows,
                sample.len(),
                if path == "eager" {
                    eager_iterations[index]
                } else {
                    graph_iterations[index]
                },
                percentile(sample, 0.50),
                percentile(sample, 0.95),
            );
        }
    }
}

fn fixed_ada_filter(name: &str, inventory: &[&str]) -> Vec<usize> {
    let requested = match std::env::var(name) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("read {name}: {error}"),
    };
    fixed_ada_direct_pair_filter(name, inventory, requested.as_deref())
        .unwrap_or_else(|error| panic!("{error}"))
}

fn fixed_ada_direct_pair_filter(
    name: &str,
    inventory: &[&str],
    requested: Option<&str>,
) -> Result<Vec<usize>, String> {
    let Some(value) = requested else {
        return Ok((0..inventory.len()).collect());
    };
    if value.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    let mut seen = std::collections::BTreeSet::new();
    value
        .split(',')
        .map(|id| {
            let index = inventory
                .iter()
                .position(|candidate| *candidate == id)
                .ok_or_else(|| format!("unknown {name} entry {id:?}; expected {inventory:?}"))?;
            if !seen.insert(index) {
                return Err(format!("duplicate {name} entry {id:?}"));
            }
            Ok(index)
        })
        .collect()
}

fn fixed_ada_direct_pair_reject_vendor_tiles(requested: Option<&str>) -> Result<(), String> {
    match requested {
        None => Ok(()),
        Some(value) => Err(format!(
            "MAMBA_FIXED_VENDOR_TILES must be absent for direct pairing, got {value:?}"
        )),
    }
}

fn fixed_ada_direct_pair_ordered_window(
    order: &str,
    mut pipeline: impl FnMut() -> f64,
    mut swizzle: impl FnMut() -> f64,
) -> Result<(f64, f64), String> {
    Ok(match order {
        "pipeline_swizzle" => (pipeline(), swizzle()),
        "swizzle_pipeline" => {
            let swizzle_us = swizzle();
            (pipeline(), swizzle_us)
        }
        _ => return Err(format!("unknown direct-pair order {order:?}")),
    })
}

fn fixed_ada_direct_pair_ratio_quantiles(
    pipeline: &[f64],
    swizzle: &[f64],
) -> ((f64, f64), (f64, f64)) {
    assert_eq!(pipeline.len(), swizzle.len(), "paired sample count");
    assert!(!pipeline.is_empty(), "paired samples must not be empty");
    let mut swizzle_over_pipeline = swizzle
        .iter()
        .zip(pipeline)
        .map(|(swizzle, pipeline)| swizzle / pipeline)
        .collect::<Vec<_>>();
    let mut pipeline_over_swizzle = pipeline
        .iter()
        .zip(swizzle)
        .map(|(pipeline, swizzle)| pipeline / swizzle)
        .collect::<Vec<_>>();
    swizzle_over_pipeline.sort_by(f64::total_cmp);
    pipeline_over_swizzle.sort_by(f64::total_cmp);
    (
        (
            percentile(&swizzle_over_pipeline, 0.5),
            percentile(&swizzle_over_pipeline, 0.95),
        ),
        (
            percentile(&pipeline_over_swizzle, 0.5),
            percentile(&pipeline_over_swizzle, 0.95),
        ),
    )
}

#[test]
fn fixed_ada_direct_pair_executes_real_windows_in_requested_order() {
    let trace = RefCell::new(Vec::new());
    let (pipeline, swizzle) = fixed_ada_direct_pair_ordered_window(
        "swizzle_pipeline",
        || {
            trace.borrow_mut().push("pipeline");
            11.0
        },
        || {
            trace.borrow_mut().push("swizzle");
            7.0
        },
    )
    .expect("valid reversed direct-pair order");
    assert_eq!((pipeline, swizzle), (11.0, 7.0));
    assert_eq!(&*trace.borrow(), &["swizzle", "pipeline"]);

    assert!(
        fixed_ada_direct_pair_ordered_window("pipeline_auto", || 1.0, || 2.0).is_err(),
        "AUTO-bearing or foreign orders must fail closed"
    );
}

#[test]
fn fixed_ada_direct_pair_quantiles_use_same_index_pairs_in_both_directions() {
    let pipeline = [100.0; 5];
    let swizzle = [50.0, 80.0, 80.0, 120.0, 120.0];
    let ((swizzle_p50, swizzle_p95), (pipeline_p50, pipeline_p95)) =
        fixed_ada_direct_pair_ratio_quantiles(&pipeline, &swizzle);
    assert_eq!((swizzle_p50, swizzle_p95), (0.8, 1.2));
    assert_eq!((pipeline_p50, pipeline_p95), (1.25, 2.0));
    assert_ne!(pipeline_p95, 1.0 / swizzle_p95);
}

#[test]
fn fixed_ada_direct_pair_filters_are_strict_and_vendor_tiles_are_forbidden() {
    assert_eq!(
        fixed_ada_direct_pair_filter("rows", &["bf16", "f16"], Some("f16,bf16"))
            .expect("valid reordered row filter"),
        vec![1, 0]
    );
    for invalid in [Some(""), Some("bf16,bf16"), Some("f32"), Some("bf16,")] {
        assert!(fixed_ada_direct_pair_filter("rows", &["bf16", "f16"], invalid).is_err());
    }
    assert!(fixed_ada_direct_pair_reject_vendor_tiles(None).is_ok());
    for supplied in [Some(""), Some("Tc128Sm89Pipeline")] {
        assert!(fixed_ada_direct_pair_reject_vendor_tiles(supplied).is_err());
    }
}

fn fixed_ada_normalized_error(
    actual: &[u32],
    reference: &[u32],
    tolerance: f64,
    label: &str,
) -> f64 {
    assert_eq!(actual.len(), reference.len(), "{label} output length");
    let mut max_error = 0.0f64;
    let mut scale = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let actual = f64::from(f32::from_bits(actual));
        let reference = f64::from(f32::from_bits(reference));
        assert!(
            actual.is_finite() && reference.is_finite(),
            "{label} non-finite output"
        );
        max_error = max_error.max((actual - reference).abs());
        scale = scale.max(reference.abs());
    }
    assert!(scale > 0.0, "{label} reference corpus is all zero");
    let normalized = max_error / scale;
    assert!(
        normalized <= tolerance,
        "{label} max absolute error / reference infinity norm {normalized} exceeds {tolerance}"
    );
    normalized
}

fn fixed_explicit_vendor_admit_cc(
    requested: Option<&str>,
    actual: (u32, u32),
) -> Result<(u32, u32), String> {
    let expected = match requested.unwrap_or("8.9") {
        "8.9" => (8, 9),
        "12.0" => (12, 0),
        value => return Err(format!("unsupported explicit vendor exact CC {value:?}")),
    };
    if actual != expected {
        return Err(format!(
            "explicit vendor exact CC {}.{} does not match actual {}.{}",
            expected.0, expected.1, actual.0, actual.1
        ));
    }
    Ok(expected)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FixedForceBiasContract {
    Either,
    NoBias,
    WithBias,
}

impl FixedForceBiasContract {
    fn allows(self, has_bias: bool) -> bool {
        match self {
            Self::Either => true,
            Self::NoBias => !has_bias,
            Self::WithBias => has_bias,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Either => "either",
            Self::NoBias => "no_bias",
            Self::WithBias => "with_bias",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedForceSpec {
    tile: InferenceTile,
    row: &'static str,
    input_dtype: WeightDtype,
    output_dtype: WeightDtype,
    compute_capability: (u32, u32),
    bias_contract: FixedForceBiasContract,
    expected_symbol: &'static str,
}

#[derive(Clone, Copy)]
struct FixedAutoPhysicalRequest<'a> {
    row: &'static str,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    selected: InferenceTile,
    compute_capability: (u32, u32),
    multiprocessors: u32,
    compiler_target: &'a str,
    state_capacity: usize,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    policy: F32TriadPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedAutoPhysicalDescriptor {
    family: InferenceTile,
    symbol: &'static str,
    storage: [WeightDtype; 3],
    backend: mamba_rs::mamba_ssm::gpu::kernel_identity::PhysicalGemmBackend,
    tile: (u32, u32),
    bk: u32,
    stages: u8,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dynamic_shared_bytes: u32,
    static_shared_bytes: u32,
    driver_abi: [(usize, usize); 5],
}

#[derive(Clone, Copy)]
struct FixedAutoRecordedRoute {
    op: ResolvedGemmOp,
    symbol: &'static str,
    dtype: PolicyDtype,
    backend: PhysicalGemmBackend,
    shape: (usize, usize, usize),
    strides: (usize, usize, usize),
    tile: (u32, u32),
    bk: u32,
    stages: u8,
    threads: u32,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dynamic_shared_bytes: u32,
}

impl From<&ResolvedGemmRoute> for FixedAutoRecordedRoute {
    fn from(route: &ResolvedGemmRoute) -> Self {
        Self {
            op: route.op,
            symbol: route.symbol,
            dtype: route.dtype,
            backend: route.backend,
            shape: route.shape,
            strides: route.strides,
            tile: route.tile,
            bk: route.bk,
            stages: route.stages,
            threads: route.threads,
            grid: route.launch.grid_dim,
            block: route.launch.block_dim,
            dynamic_shared_bytes: route.launch.shared_mem_bytes,
        }
    }
}

#[derive(Clone)]
struct FixedAutoGraphObservation<'a> {
    node_count: usize,
    symbol: &'a str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dynamic_shared_bytes: u32,
    static_shared_bytes: u32,
    driver_abi: Vec<(usize, usize)>,
    terminal_sixth_rejected: bool,
    pointers: [u64; 4],
    bundle: [u32; 8],
}

fn fixed_auto_bundle_physical_descriptor(
    request: FixedAutoPhysicalRequest<'_>,
    recorded: &[FixedAutoRecordedRoute],
) -> Result<Option<FixedAutoPhysicalDescriptor>, String> {
    let FixedAutoPhysicalRequest {
        row,
        operands,
        shape,
        selected,
        compute_capability,
        multiprocessors,
        compiler_target,
        state_capacity,
        nvrtc,
        nvrtc_library_known,
        policy,
    } = request;
    let pointers_admitted = [operands.c.ptr, operands.x.ptr, operands.w.ptr]
        .into_iter()
        .all(|pointer| pointer != 0 && pointer.is_multiple_of(16))
        && operands
            .bias_ptr
            .is_none_or(|pointer| pointer != 0 && pointer.is_multiple_of(4));
    if compute_capability != (8, 9)
        || multiprocessors != 142
        || compiler_target != "sm_89"
        || !matches!(state_capacity, 16 | 64)
        || !matches!(nvrtc, (12, 8) | (13, 0) | (13, 2))
        || !nvrtc_library_known
        || !pointers_admitted
        || policy != F32TriadPolicy::ExactScalarFma
    {
        return Ok(None);
    }

    let driver_abi = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
    let descriptor = match (
        row,
        operands.x.dtype,
        operands.w.dtype,
        operands.c.dtype,
        (shape.m, shape.k, shape.n),
        operands.bias_ptr.is_some(),
    ) {
        (
            "bf16_f32",
            WeightDtype::Bf16,
            WeightDtype::Bf16,
            WeightDtype::F32,
            (4621, 768, 2304),
            _,
        ) => FixedAutoPhysicalDescriptor {
            family: InferenceTile::Tc128Sm89S3,
            symbol: "nn_sm89_tc128_f32out_s3_bf16",
            storage: [WeightDtype::Bf16, WeightDtype::Bf16, WeightDtype::F32],
            backend: PhysicalGemmBackend::InferenceMma16,
            tile: (128, 128),
            bk: 64,
            stages: 3,
            grid: (666, 1, 1),
            block: (256, 1, 1),
            dynamic_shared_bytes: 98_304,
            static_shared_bytes: 0,
            driver_abi,
        },
        (
            "bf16_f32",
            WeightDtype::Bf16,
            WeightDtype::Bf16,
            WeightDtype::F32,
            (4621, 1928, 384),
            _,
        ) => FixedAutoPhysicalDescriptor {
            family: InferenceTile::Tc128Sm89S3,
            symbol: "nn_sm89_tc128_f32out_s3_bf16",
            storage: [WeightDtype::Bf16, WeightDtype::Bf16, WeightDtype::F32],
            backend: PhysicalGemmBackend::InferenceMma16,
            tile: (128, 128),
            bk: 64,
            stages: 3,
            grid: (111, 1, 1),
            block: (256, 1, 1),
            dynamic_shared_bytes: 98_304,
            static_shared_bytes: 0,
            driver_abi,
        },
        ("f16_f32", WeightDtype::F16, WeightDtype::F16, WeightDtype::F32, (4621, 768, 2304), _) => {
            FixedAutoPhysicalDescriptor {
                family: InferenceTile::Tc128Sm89S3,
                symbol: "nn_sm89_tc128_f32out_s3_f16",
                storage: [WeightDtype::F16, WeightDtype::F16, WeightDtype::F32],
                backend: PhysicalGemmBackend::InferenceMma16,
                tile: (128, 128),
                bk: 64,
                stages: 3,
                grid: (666, 1, 1),
                block: (256, 1, 1),
                dynamic_shared_bytes: 98_304,
                static_shared_bytes: 0,
                driver_abi,
            }
        }
        ("f16_f32", WeightDtype::F16, WeightDtype::F16, WeightDtype::F32, (4621, 1928, 384), _) => {
            FixedAutoPhysicalDescriptor {
                family: InferenceTile::Tc128Sm89S3,
                symbol: "nn_sm89_tc128_f32out_s3_f16",
                storage: [WeightDtype::F16, WeightDtype::F16, WeightDtype::F32],
                backend: PhysicalGemmBackend::InferenceMma16,
                tile: (128, 128),
                bk: 64,
                stages: 3,
                grid: (111, 1, 1),
                block: (256, 1, 1),
                dynamic_shared_bytes: 98_304,
                static_shared_bytes: 0,
                driver_abi,
            }
        }
        (
            "f32_exact" | "f32_exact_fast",
            WeightDtype::F32,
            WeightDtype::F32,
            WeightDtype::F32,
            (4621, 1928, 384),
            false,
        ) => FixedAutoPhysicalDescriptor {
            family: InferenceTile::F32Sm89N64CopyPlan,
            symbol: "nn_sm89_f32_m128n64_tail_copyplan",
            storage: [WeightDtype::F32; 3],
            backend: PhysicalGemmBackend::InferenceScalarFma,
            tile: (128, 64),
            bk: 32,
            stages: 2,
            grid: (222, 1, 1),
            block: (256, 1, 1),
            dynamic_shared_bytes: 0,
            static_shared_bytes: 49_152,
            driver_abi,
        },
        _ => return Ok(None),
    };

    if selected != descriptor.family {
        return Err(format!(
            "fully supported retained Inference AUTO request expected admitted holder {} and family {:?}, but AUTO returned {selected:?}; the holder may have been rejected and AUTO fell back",
            descriptor.symbol, descriptor.family
        ));
    }
    let [actual] = recorded else {
        return Err(format!(
            "fully supported retained Inference AUTO request expected exactly one recorded holder route {}, got {}",
            descriptor.symbol,
            recorded.len()
        ));
    };
    let expected_dtype = match descriptor.storage[0] {
        WeightDtype::F32 => PolicyDtype::F32,
        WeightDtype::F16 => PolicyDtype::F16,
        WeightDtype::Bf16 => PolicyDtype::Bf16,
    };
    if actual.op != ResolvedGemmOp::Nn
        || actual.symbol != descriptor.symbol
        || actual.dtype != expected_dtype
        || actual.backend != descriptor.backend
        || actual.shape != (shape.m, shape.k, shape.n)
        || actual.strides != (shape.k, shape.n, shape.n)
        || actual.tile != descriptor.tile
        || actual.bk != descriptor.bk
        || actual.stages != descriptor.stages
        || actual.threads != descriptor.block.0
        || actual.grid != descriptor.grid
        || actual.block != descriptor.block
        || actual.dynamic_shared_bytes != descriptor.dynamic_shared_bytes
    {
        return Err(format!(
            "fully supported retained Inference AUTO request expected admitted holder route {}, but recorded symbol {:?}, dtype {:?}, backend {:?}, shape {:?}, strides {:?}, tile {:?}, bk {}, stages {}, threads {}, grid {:?}, block {:?}, dynamic shared {}; the holder may have been rejected and AUTO fell back",
            descriptor.symbol,
            actual.symbol,
            actual.dtype,
            actual.backend,
            actual.shape,
            actual.strides,
            actual.tile,
            actual.bk,
            actual.stages,
            actual.threads,
            actual.grid,
            actual.block,
            actual.dynamic_shared_bytes
        ));
    }
    Ok(Some(descriptor))
}

fn fixed_auto_bundle_graph_contract(
    descriptor: FixedAutoPhysicalDescriptor,
    observed: &FixedAutoGraphObservation<'_>,
    expected_pointers: [u64; 4],
    shape: InferenceShape,
) -> Result<(), String> {
    let expected_bundle = [
        1.0f32.to_bits(),
        0.0f32.to_bits(),
        shape.m as u32,
        shape.n as u32,
        shape.k as u32,
        shape.k as u32,
        shape.n as u32,
        shape.n as u32,
    ];
    if observed.node_count != 1
        || observed.symbol != descriptor.symbol
        || observed.grid != descriptor.grid
        || observed.block != descriptor.block
        || observed.dynamic_shared_bytes != descriptor.dynamic_shared_bytes
        || observed.static_shared_bytes != descriptor.static_shared_bytes
        || observed.driver_abi != descriptor.driver_abi
        || !observed.terminal_sixth_rejected
        || observed.pointers != expected_pointers
        || observed.bundle != expected_bundle
    {
        return Err(format!(
            "wrong retained Inference AUTO physical graph: nodes={} symbol={:?} grid={:?} block={:?} dynamic_shared={} static_shared={} abi={:?} sixth_rejected={} pointers={:?} expected_pointers={expected_pointers:?} bundle={:?}",
            observed.node_count,
            observed.symbol,
            observed.grid,
            observed.block,
            observed.dynamic_shared_bytes,
            observed.static_shared_bytes,
            observed.driver_abi,
            observed.terminal_sixth_rejected,
            observed.pointers,
            observed.bundle
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct FixedExplicitVendorRowSpec {
    name: &'static str,
    input_dtype: WeightDtype,
    output_dtype: WeightDtype,
    policy: F32TriadPolicy,
    vendor_compute: cudarc::cublas::sys::cublasComputeType_t,
    vendor_comparator: &'static str,
    custom_tolerance: f64,
    vendor_tolerance: f64,
}

const FIXED_PRODUCTION_AUTO_DEFAULT_WINDOWS: usize = 21;

fn fixed_production_auto_expected_records(
    rows: usize,
    cells: usize,
    biases: usize,
    paths: usize,
) -> usize {
    rows * cells * biases * paths * 2
}

fn fixed_production_auto_ratio_samples(
    auto_us: &[f64],
    vendor_us: &[f64],
) -> Result<Vec<f64>, String> {
    if auto_us.len() != vendor_us.len() || auto_us.is_empty() {
        return Err("production AUTO ratio arms require equal non-empty sample counts".into());
    }
    auto_us
        .iter()
        .zip(vendor_us)
        .enumerate()
        .map(|(index, (&auto, &vendor))| {
            if !auto.is_finite() || auto <= 0.0 || !vendor.is_finite() || vendor <= 0.0 {
                return Err(format!(
                    "production AUTO pair {index} is not positive and finite"
                ));
            }
            let ratio = auto / vendor;
            if !ratio.is_finite() || ratio <= 0.0 {
                return Err(format!(
                    "production AUTO/vendor ratio {index} is not positive and finite"
                ));
            }
            Ok(ratio)
        })
        .collect()
}

#[test]
fn production_auto_state_capacity_parser_is_strict() {
    use std::ffi::OsStr;

    assert_eq!(production_auto_cohort::parse_state_capacity(None), Ok(64));
    assert_eq!(
        production_auto_cohort::parse_state_capacity(Some(OsStr::new("16"))),
        Ok(16)
    );
    assert_eq!(
        production_auto_cohort::parse_state_capacity(Some(OsStr::new("64"))),
        Ok(64)
    );
    for invalid in ["", "016", "32", " 16", "64 ", "16\n"] {
        assert!(
            production_auto_cohort::parse_state_capacity(Some(OsStr::new(invalid))).is_err(),
            "accepted invalid state capacity {invalid:?}"
        );
    }
}

#[cfg(unix)]
#[test]
fn production_auto_state_capacity_parser_rejects_non_utf8() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt as _;

    assert!(
        production_auto_cohort::parse_state_capacity(Some(OsStr::from_bytes(&[b'1', 0xff])))
            .is_err()
    );
}

#[test]
fn production_auto_cohort_fragment_is_valid_json_with_actual_resources() {
    let fragment = production_auto_cohort::render_cohort_fragment(
        production_auto_cohort::ProductionAutoInventory::Release070,
        16,
        12_345,
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&format!("{{{fragment}}}")).expect("valid cohort JSON fragment");

    assert_eq!(parsed["inventory"], "release_070");
    assert_eq!(parsed["state_capacity"], 16);
    assert_eq!(parsed["cublas_workspace_bytes"], 12_345);

    let outlier_fragment = production_auto_cohort::render_cohort_fragment(
        production_auto_cohort::ProductionAutoInventory::Outliers071,
        64,
        8_192,
    );
    let outlier: serde_json::Value = serde_json::from_str(&format!("{{{outlier_fragment}}}"))
        .expect("valid outlier cohort JSON fragment");
    assert_eq!(outlier["inventory"], "outliers_071");
    assert_eq!(outlier["state_capacity"], 64);
    assert_eq!(outlier["cublas_workspace_bytes"], 8_192);
    assert!(
        production_auto_cohort::validate_completion_counts(
            production_auto_cohort::ProductionAutoInventory::Release070,
            66,
            81,
            324,
        )
        .is_ok()
    );
    assert!(
        production_auto_cohort::validate_completion_counts(
            production_auto_cohort::ProductionAutoInventory::Outliers071,
            72,
            93,
            372,
        )
        .is_ok()
    );
}

fn fixed_production_auto_inventory_has_exact_symbol(inventory: &str, expected: &str) -> bool {
    inventory.contains(&format!(
        "\"symbol\":\"{}\"",
        fixed_sm120_tf32_bd_json_escape(expected)
    ))
}

fn fixed_production_auto_format_cuda_uuid(bytes: [std::ffi::c_char; 16]) -> String {
    let bytes = bytes.map(|byte| byte as u8);
    format!(
        concat!(
            "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-",
            "{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}"
        ),
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
    )
}

fn fixed_production_auto_cuda_uuid(cuda_ordinal: usize) -> Result<String, String> {
    cudarc::driver::result::init().map_err(|error| format!("initialize CUDA driver: {error}"))?;
    let ordinal = i32::try_from(cuda_ordinal)
        .map_err(|_| format!("CUDA ordinal {cuda_ordinal} exceeds i32::MAX"))?;
    let device = cudarc::driver::result::device::get(ordinal)
        .map_err(|error| format!("resolve CUDA ordinal {cuda_ordinal}: {error}"))?;
    let uuid = cudarc::driver::result::device::get_uuid(device)
        .map_err(|error| format!("query CUDA ordinal {cuda_ordinal} UUID: {error}"))?;
    Ok(fixed_production_auto_format_cuda_uuid(uuid.bytes))
}

fn fixed_production_auto_tuning_metadata() -> String {
    format!("\"tuning_table_revision\":{TUNING_TABLE_REVISION}")
}

fn fixed_explicit_vendor_row_specs() -> [FixedExplicitVendorRowSpec; 7] {
    use cudarc::cublas::sys::cublasComputeType_t;

    [
        FixedExplicitVendorRowSpec {
            name: "bf16",
            input_dtype: WeightDtype::Bf16,
            output_dtype: WeightDtype::Bf16,
            policy: F32TriadPolicy::ExactScalarFma,
            vendor_compute: cublasComputeType_t::CUBLAS_COMPUTE_32F,
            vendor_comparator: "CUBLAS_COMPUTE_32F",
            custom_tolerance: 0.01,
            vendor_tolerance: 0.01,
        },
        FixedExplicitVendorRowSpec {
            name: "f16",
            input_dtype: WeightDtype::F16,
            output_dtype: WeightDtype::F16,
            policy: F32TriadPolicy::ExactScalarFma,
            vendor_compute: cublasComputeType_t::CUBLAS_COMPUTE_32F,
            vendor_comparator: "CUBLAS_COMPUTE_32F",
            custom_tolerance: 0.0025,
            vendor_tolerance: 0.0025,
        },
        FixedExplicitVendorRowSpec {
            name: "bf16_f32",
            input_dtype: WeightDtype::Bf16,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::ExactScalarFma,
            vendor_compute: cublasComputeType_t::CUBLAS_COMPUTE_32F,
            vendor_comparator: "CUBLAS_COMPUTE_32F",
            custom_tolerance: 0.01,
            vendor_tolerance: 0.01,
        },
        FixedExplicitVendorRowSpec {
            name: "f16_f32",
            input_dtype: WeightDtype::F16,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::ExactScalarFma,
            vendor_compute: cublasComputeType_t::CUBLAS_COMPUTE_32F,
            vendor_comparator: "CUBLAS_COMPUTE_32F",
            custom_tolerance: 0.0025,
            vendor_tolerance: 0.0025,
        },
        FixedExplicitVendorRowSpec {
            name: "tf32",
            input_dtype: WeightDtype::F32,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::AllowDeterministicTf32,
            vendor_compute: cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            vendor_comparator: "CUBLAS_COMPUTE_32F_FAST_TF32",
            custom_tolerance: 0.0025,
            vendor_tolerance: 0.0025,
        },
        FixedExplicitVendorRowSpec {
            name: "f32_exact",
            input_dtype: WeightDtype::F32,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::ExactScalarFma,
            vendor_compute: cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
            vendor_comparator: "CUBLAS_COMPUTE_32F_PEDANTIC",
            custom_tolerance: 0.0002,
            vendor_tolerance: 0.0002,
        },
        FixedExplicitVendorRowSpec {
            name: "f32_exact_fast",
            input_dtype: WeightDtype::F32,
            output_dtype: WeightDtype::F32,
            policy: F32TriadPolicy::ExactScalarFma,
            vendor_compute: cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            vendor_comparator: "CUBLAS_COMPUTE_32F_FAST_TF32",
            custom_tolerance: 0.0002,
            vendor_tolerance: 0.0025,
        },
    ]
}

fn fixed_exact_comparator_completion_metadata() -> String {
    let rows = fixed_explicit_vendor_row_specs();
    let comparator = |name| {
        rows.iter()
            .find(|row| row.name == name)
            .unwrap_or_else(|| panic!("missing exact comparator row {name}"))
            .vendor_comparator
    };
    format!(
        "\"exact_comparators\":{{\"f32_exact\":\"{}\",\"f32_exact_fast\":\"{}\"}}",
        comparator("f32_exact"),
        comparator("f32_exact_fast")
    )
}

fn fixed_explicit_vendor_tolerance_metadata(row: &FixedExplicitVendorRowSpec) -> String {
    format!(
        "\"vendor_comparator\":\"{}\",\"normalized_error_tolerance\":{},\"custom_normalized_error_tolerance\":{},\"vendor_normalized_error_tolerance\":{}",
        row.vendor_comparator, row.custom_tolerance, row.custom_tolerance, row.vendor_tolerance,
    )
}

// Each registry macro drives both the iterable inventory universe and an
// exhaustive match. Adding an enum variant therefore cannot compile until it
// is classified here, and classification automatically adds it to the reverse
// inventory check.
macro_rules! define_fixed_force_half_tile_registry {
    ($($variant:ident),+ $(,)?) => {
        const FIXED_FORCE_HALF_TILE_UNIVERSE: &[InferenceSm120HalfTile] = &[
            $(InferenceSm120HalfTile::$variant),+
        ];

        fn fixed_force_half_tile_registry_exhaustive(tile: InferenceSm120HalfTile) {
            match tile {
                $(InferenceSm120HalfTile::$variant => {}),+
            }
        }
    };
}

define_fixed_force_half_tile_registry!(
    M64N64Bk64S2,
    M64N128Bk64S2,
    M128N64Bk32S3,
    M128N128Bk32S2,
    M128N128Bk32S3,
);

macro_rules! define_fixed_force_plain_tile_registry {
    ($($variant:ident),+ $(,)?) => {
        const FIXED_FORCE_PLAIN_TILE_UNIVERSE: &[InferenceTile] = &[
            $(InferenceTile::$variant),+
        ];

        fn fixed_force_tile_registry_exhaustive(tile: InferenceTile) {
            match tile {
                $(InferenceTile::$variant => {}),+,
                InferenceTile::Sm120Half(half) => fixed_force_half_tile_registry_exhaustive(half),
            }
        }
    };
}

#[test]
fn fixed_force_registry_contains_selectable_ada_s3() {
    assert!(
        fixed_force_tile_universe()
            .iter()
            .any(|tile| format!("{tile:?}") == "Tc128Sm89S3"),
        "the public force universe must include the S3 route"
    );
}

#[test]
fn fixed_force_registry_and_inventory_include_all_three_ada_finalists() {
    for (name, row, symbol, bias_contract) in [
        (
            "Tf32RnaM128N96S3",
            "tf32",
            "nn_sm89_rna_tf32_m128n96_bk32_s3",
            FixedForceBiasContract::Either,
        ),
        (
            "TcM64N64Sm89S3",
            "f16",
            "nn_sm89_m64n64_bk64_s3_f16",
            FixedForceBiasContract::NoBias,
        ),
        (
            "TcM128N64Sm89S2",
            "f16",
            "nn_sm89_m128n64_bk64_s2_f16",
            FixedForceBiasContract::NoBias,
        ),
    ] {
        assert!(
            fixed_force_tile_universe()
                .iter()
                .any(|tile| format!("{tile:?}") == name),
            "force registry omitted {name}"
        );
        let selected = fixed_explicit_vendor_filter_tiles(
            &fixed_explicit_vendor_tiles(row, (8, 9)),
            Some(name),
        )
        .expect("finalist is selectable from the Ada force inventory");
        assert_eq!(selected.len(), 1);
        let spec = fixed_force_spec(row, (8, 9), selected[0]).expect("physical finalist spec");
        assert_eq!(spec.expected_symbol, symbol);
        assert_eq!(spec.bias_contract, bias_contract);
        assert!(fixed_explicit_vendor_needs_identity_graph(
            &["eager"],
            selected[0]
        ));
        for bad_row in [
            "bf16",
            "f16",
            "tf32",
            "bf16_f32",
            "f16_f32",
            "f32_exact",
            "f32_exact_fast",
        ] {
            if bad_row != row {
                assert!(fixed_force_spec(bad_row, (8, 9), selected[0]).is_err());
            }
        }
        assert!(
            fixed_explicit_vendor_filter_tiles(
                &fixed_explicit_vendor_tiles(row, (12, 0)),
                Some(name),
            )
            .is_err()
        );
        assert!(fixed_force_spec(row, (12, 0), selected[0]).is_err());
    }
}

define_fixed_force_plain_tile_registry!(
    F32N128S2,
    Tf32M128S2,
    Tf32M128S3,
    Tf32M128N128S3,
    Tf32RnaM128N128S3,
    Tf32RnaM128N96S3,
    Tf32M64S2,
    Tf32M64S3,
    Tf32M16S4,
    Tf32Sm120M128S2,
    Tf32Sm120M128S3,
    Tf32Sm120M64N128S2,
    Tf32Sm120M64N128S3,
    Tf32Sm120M64S2ProducerWarp,
    Tf32Sm120M64S2,
    Tf32Sm120M64S2PairStore,
    Tc128,
    Tc128Sm89Pipeline,
    Tc128Sm89Swizzle,
    Tc128Sm89S3,
    TcM64N64Sm89S3,
    TcM128N64Sm89S2,
    TcWn64,
    TcW64,
    Tc64,
    Tc16,
    Legacy,
    Sm90Wgmma,
    Sm100Tcgen,
    F32Sm89N64CopyPlan,
    F32Sm120N64CopyPlan,
    F32Sm120N64CopyPlanT256,
    F32Sm120M128N64CopyPlanT256,
    F32Sm120N64Sliced,
    F32Sm120TmaFmaM128N64,
    F32Sm120TmaFmaM64N128,
    F32Sm120TmaFmaFixedPostBiasM128N64,
    F32Sm120TmaFmaFixedPostBiasM64N128,
    F32Sm120TmaFmaFixedPostBiasM128N96,
    F32Sm120TmaFmaFixedPostBiasM128N64K4,
    F32Sm120TmaFmaFixedPostBiasM128N64T256,
    F32Sm120TmaFmaFixedNoBiasM128N64T256,
);

fn fixed_force_tile_universe() -> Vec<InferenceTile> {
    let mut universe = FIXED_FORCE_PLAIN_TILE_UNIVERSE.to_vec();
    universe.extend(
        FIXED_FORCE_HALF_TILE_UNIVERSE
            .iter()
            .copied()
            .map(InferenceTile::Sm120Half),
    );
    for tile in universe.iter().copied() {
        fixed_force_tile_registry_exhaustive(tile);
    }
    universe
}

#[derive(Debug, PartialEq, Eq)]
enum FixedForceFirstLaunch {
    Runnable,
    BiasContractRejected {
        contract: &'static str,
        reason: String,
    },
    AvailabilityRejected(String),
}

fn fixed_force_classify_first_launch(
    spec: &FixedForceSpec,
    has_bias: bool,
    launch: Result<(), String>,
) -> Result<FixedForceFirstLaunch, String> {
    if !spec.bias_contract.allows(has_bias) {
        return match launch {
            Err(reason) => Ok(FixedForceFirstLaunch::BiasContractRejected {
                contract: spec.bias_contract.label(),
                reason,
            }),
            Ok(()) => Err(format!(
                "{:?} accepted disallowed bias={has_bias} for {} contract",
                spec.tile,
                spec.bias_contract.label()
            )),
        };
    }
    Ok(match launch {
        Ok(()) => FixedForceFirstLaunch::Runnable,
        Err(reason) => FixedForceFirstLaunch::AvailabilityRejected(reason),
    })
}

fn fixed_force_run_first_launch(
    spec: &FixedForceSpec,
    has_bias: bool,
    launch: impl FnOnce() -> Result<(), String>,
) -> Result<FixedForceFirstLaunch, String> {
    fixed_force_classify_first_launch(spec, has_bias, launch())
}

fn fixed_force_row_dtypes(row: &str) -> Result<(WeightDtype, WeightDtype), String> {
    match row {
        "bf16" => Ok((WeightDtype::Bf16, WeightDtype::Bf16)),
        "f16" => Ok((WeightDtype::F16, WeightDtype::F16)),
        "bf16_f32" => Ok((WeightDtype::Bf16, WeightDtype::F32)),
        "f16_f32" => Ok((WeightDtype::F16, WeightDtype::F32)),
        "tf32" | "f32_exact" | "f32_exact_fast" => Ok((WeightDtype::F32, WeightDtype::F32)),
        _ => Err(format!("unsupported explicit vendor row {row:?}")),
    }
}

// Keep this match exhaustive: adding a InferenceTile without deciding its physical
// force identity must fail compilation instead of silently shrinking the inventory.
fn fixed_force_spec(
    row: &'static str,
    cc: (u32, u32),
    tile: InferenceTile,
) -> Result<FixedForceSpec, String> {
    let (input_dtype, output_dtype) = fixed_force_row_dtypes(row)?;
    let invalid = || format!("{tile:?} is not a physical Fixed force candidate for {row}/CC{cc:?}");
    let expected_symbol = match tile {
        InferenceTile::F32N128S2 if matches!(row, "f32_exact" | "f32_exact_fast") => {
            "f32_f32_n128_s2"
        }
        InferenceTile::F32N128S2 => return Err(invalid()),
        InferenceTile::Tf32M128S2 if row == "tf32" => "nn_tf32_m128n64_bk32_s2",
        InferenceTile::Tf32M128S2 => return Err(invalid()),
        InferenceTile::Tf32M128S3 if row == "tf32" => "nn_tf32_m128n64_bk32_s3",
        InferenceTile::Tf32M128S3 => return Err(invalid()),
        InferenceTile::Tf32M128N128S3 if row == "tf32" && cc == (8, 9) => {
            "nn_sm80_mma_tf32_m128n128_bk32_s3"
        }
        InferenceTile::Tf32M128N128S3 => return Err(invalid()),
        InferenceTile::Tf32RnaM128N128S3 if row == "tf32" && cc == (8, 9) => {
            "nn_rna_wide_tf32_m128n128_bk32_s3"
        }
        InferenceTile::Tf32RnaM128N128S3 => return Err(invalid()),
        InferenceTile::Tf32RnaM128N96S3 if row == "tf32" && cc == (8, 9) => {
            "nn_sm89_rna_tf32_m128n96_bk32_s3"
        }
        InferenceTile::Tf32RnaM128N96S3 => return Err(invalid()),
        InferenceTile::Tf32M64S2 if row == "tf32" => "nn_tf32_m64n64_bk32_s2",
        InferenceTile::Tf32M64S2 => return Err(invalid()),
        InferenceTile::Tf32M64S3 if row == "tf32" => "nn_tf32_m64n64_bk32_s3",
        InferenceTile::Tf32M64S3 => return Err(invalid()),
        InferenceTile::Tf32M16S4 if row == "tf32" => "nn_tf32_m16n32_bk32_s4",
        InferenceTile::Tf32M16S4 => return Err(invalid()),
        InferenceTile::Tf32Sm120M128S2 if row == "tf32" && cc == (12, 0) => {
            "nn_sm120_tma_tf32_m128n64_bk32_s2"
        }
        InferenceTile::Tf32Sm120M128S2 => return Err(invalid()),
        InferenceTile::Tf32Sm120M128S3 if row == "tf32" && cc == (12, 0) => {
            "nn_sm120_tma_tf32_m128n64_bk32_s3"
        }
        InferenceTile::Tf32Sm120M128S3 => return Err(invalid()),
        InferenceTile::Tf32Sm120M64N128S2 if row == "tf32" && cc == (12, 0) => {
            "nn_sm120_tma_tf32_m64n128_bk32_s2"
        }
        InferenceTile::Tf32Sm120M64N128S2 => return Err(invalid()),
        InferenceTile::Tf32Sm120M64N128S3 if row == "tf32" && cc == (12, 0) => {
            "nn_sm120_tma_tf32_m64n128_bk32_s3"
        }
        InferenceTile::Tf32Sm120M64N128S3 => return Err(invalid()),
        InferenceTile::Tf32Sm120M64S2ProducerWarp if row == "tf32" && cc == (12, 0) => {
            "nn_sm120_tma_tf32_m64n64_bk32_s2_producer_warp"
        }
        InferenceTile::Tf32Sm120M64S2ProducerWarp => return Err(invalid()),
        InferenceTile::Tf32Sm120M64S2 if row == "tf32" && cc == (12, 0) => {
            "nn_sm120_tma_tf32_m64n64_bk32_s2"
        }
        InferenceTile::Tf32Sm120M64S2 => return Err(invalid()),
        InferenceTile::Tf32Sm120M64S2PairStore if row == "tf32" && cc == (12, 0) => {
            "nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store"
        }
        InferenceTile::Tf32Sm120M64S2PairStore => return Err(invalid()),
        InferenceTile::Sm120Half(half) if cc == (12, 0) => match (half, row) {
            (InferenceSm120HalfTile::M64N64Bk64S2, "bf16") => "nn_sm120_tma_64x64_bk64_s2_bf16",
            (InferenceSm120HalfTile::M64N64Bk64S2, "f16") => "nn_sm120_tma_64x64_bk64_s2_f16",
            (InferenceSm120HalfTile::M64N64Bk64S2, "bf16_f32") => {
                "nn_sm120_tma_64x64_bk64_s2_f32out_bf16"
            }
            (InferenceSm120HalfTile::M64N64Bk64S2, "f16_f32") => {
                "nn_sm120_tma_64x64_bk64_s2_f32out_f16"
            }
            (InferenceSm120HalfTile::M64N128Bk64S2, "bf16") => "nn_sm120_tma_64x128_bk64_s2_bf16",
            (InferenceSm120HalfTile::M64N128Bk64S2, "f16") => "nn_sm120_tma_64x128_bk64_s2_f16",
            (InferenceSm120HalfTile::M64N128Bk64S2, "bf16_f32") => {
                "nn_sm120_tma_64x128_bk64_s2_f32out_bf16"
            }
            (InferenceSm120HalfTile::M64N128Bk64S2, "f16_f32") => {
                "nn_sm120_tma_64x128_bk64_s2_f32out_f16"
            }
            (InferenceSm120HalfTile::M128N64Bk32S3, "bf16") => "nn_sm120_tma_128x64_bk32_s3_bf16",
            (InferenceSm120HalfTile::M128N64Bk32S3, "f16") => "nn_sm120_tma_128x64_bk32_s3_f16",
            (InferenceSm120HalfTile::M128N64Bk32S3, "bf16_f32") => {
                "nn_sm120_tma_128x64_bk32_s3_f32out_bf16"
            }
            (InferenceSm120HalfTile::M128N64Bk32S3, "f16_f32") => {
                "nn_sm120_tma_128x64_bk32_s3_f32out_f16"
            }
            (InferenceSm120HalfTile::M128N128Bk32S2, "bf16") => "nn_sm120_tma_128x128_bk32_s2_bf16",
            (InferenceSm120HalfTile::M128N128Bk32S2, "f16") => "nn_sm120_tma_128x128_bk32_s2_f16",
            (InferenceSm120HalfTile::M128N128Bk32S2, "bf16_f32") => {
                "nn_sm120_tma_128x128_bk32_s2_f32out_bf16"
            }
            (InferenceSm120HalfTile::M128N128Bk32S2, "f16_f32") => {
                "nn_sm120_tma_128x128_bk32_s2_f32out_f16"
            }
            (InferenceSm120HalfTile::M128N128Bk32S3, "bf16") => "nn_sm120_tma_128x128_bk32_s3_bf16",
            (InferenceSm120HalfTile::M128N128Bk32S3, "f16") => "nn_sm120_tma_128x128_bk32_s3_f16",
            (InferenceSm120HalfTile::M128N128Bk32S3, "bf16_f32") => {
                "nn_sm120_tma_128x128_bk32_s3_f32out_bf16"
            }
            (InferenceSm120HalfTile::M128N128Bk32S3, "f16_f32") => {
                "nn_sm120_tma_128x128_bk32_s3_f32out_f16"
            }
            _ => return Err(invalid()),
        },
        InferenceTile::Sm120Half(_) => return Err(invalid()),
        InferenceTile::Tc128 if matches!(row, "bf16" | "f16" | "bf16_f32" | "f16_f32") => match row
        {
            "bf16" => "nn_tc128_bf16",
            "f16" => "nn_tc128_f16",
            "bf16_f32" => "nn_tc128_f32out_bf16",
            "f16_f32" => "nn_tc128_f32out_f16",
            _ => unreachable!(),
        },
        InferenceTile::Tc128 => return Err(invalid()),
        InferenceTile::Tc128Sm89Pipeline if cc == (8, 9) && row == "bf16" => {
            "nn_sm89_tc128_pipeline_bf16"
        }
        InferenceTile::Tc128Sm89Pipeline if cc == (8, 9) && row == "f16" => {
            "nn_sm89_tc128_pipeline_f16"
        }
        InferenceTile::Tc128Sm89Pipeline => return Err(invalid()),
        InferenceTile::Tc128Sm89Swizzle if cc == (8, 9) && row == "bf16" => {
            "nn_sm89_tc128_swizzle_bf16"
        }
        InferenceTile::Tc128Sm89Swizzle if cc == (8, 9) && row == "f16" => {
            "nn_sm89_tc128_swizzle_f16"
        }
        InferenceTile::Tc128Sm89Swizzle => return Err(invalid()),
        InferenceTile::Tc128Sm89S3 if cc == (8, 9) && row == "bf16" => "nn_sm89_tc128_s3_bf16",
        InferenceTile::Tc128Sm89S3 if cc == (8, 9) && row == "f16" => "nn_sm89_tc128_s3_f16",
        InferenceTile::Tc128Sm89S3 => return Err(invalid()),
        InferenceTile::TcM64N64Sm89S3 if cc == (8, 9) && row == "f16" => {
            "nn_sm89_m64n64_bk64_s3_f16"
        }
        InferenceTile::TcM128N64Sm89S2 if cc == (8, 9) && row == "f16" => {
            "nn_sm89_m128n64_bk64_s2_f16"
        }
        InferenceTile::TcM64N64Sm89S3 | InferenceTile::TcM128N64Sm89S2 => return Err(invalid()),
        InferenceTile::TcWn64 if row == "bf16" => "nn_tcwn64_bf16",
        InferenceTile::TcWn64 if row == "f16" => "nn_tcwn64_f16",
        InferenceTile::TcWn64 => return Err(invalid()),
        InferenceTile::TcW64 if row == "bf16" => "nn_tcw64_bf16",
        InferenceTile::TcW64 if row == "f16" => "nn_tcw64_f16",
        InferenceTile::TcW64 => return Err(invalid()),
        InferenceTile::Tc64 if matches!(row, "bf16" | "f16" | "bf16_f32" | "f16_f32") => {
            match row {
                "bf16" => "nn_tc64_bf16",
                "f16" => "nn_tc64_f16",
                "bf16_f32" => "nn_tc64_f32out_bf16",
                "f16_f32" => "nn_tc64_f32out_f16",
                _ => unreachable!(),
            }
        }
        InferenceTile::Tc64 => return Err(invalid()),
        InferenceTile::Tc16 if matches!(row, "bf16" | "f16" | "bf16_f32" | "f16_f32") => {
            match row {
                "bf16" => "nn_tc16_bf16",
                "f16" => "nn_tc16_f16",
                "bf16_f32" => "nn_tc16_f32out_bf16",
                "f16_f32" => "nn_tc16_f32out_f16",
                _ => unreachable!(),
            }
        }
        InferenceTile::Tc16 => return Err(invalid()),
        InferenceTile::Legacy => match row {
            "bf16" => "bf16_bf16",
            "f16" => "f16_f16",
            "bf16_f32" => "bf16_f32",
            "f16_f32" => "f16_f32",
            "f32_exact" | "f32_exact_fast" => "f32_f32_s2",
            _ => return Err(invalid()),
        },
        InferenceTile::Sm90Wgmma | InferenceTile::Sm100Tcgen => return Err(invalid()),
        InferenceTile::F32Sm89N64CopyPlan
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (8, 9) =>
        {
            "nn_sm89_f32_n64_copyplan"
        }
        InferenceTile::F32Sm89N64CopyPlan => return Err(invalid()),
        InferenceTile::F32Sm120N64CopyPlan
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (12, 0) =>
        {
            "nn_sm120_f32_n64_copyplan"
        }
        InferenceTile::F32Sm120N64CopyPlan => return Err(invalid()),
        InferenceTile::F32Sm120N64CopyPlanT256
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (12, 0) =>
        {
            "nn_sm120_f32_n64_copyplan_t256"
        }
        InferenceTile::F32Sm120N64CopyPlanT256 => return Err(invalid()),
        InferenceTile::F32Sm120M128N64CopyPlanT256
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (12, 0) =>
        {
            "nn_sm120_f32_n64_copyplan_m128n64_t256"
        }
        InferenceTile::F32Sm120M128N64CopyPlanT256 => return Err(invalid()),
        InferenceTile::F32Sm120N64Sliced
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (12, 0) =>
        {
            "nn_sm120_f32_n64_sliced"
        }
        InferenceTile::F32Sm120N64Sliced => return Err(invalid()),
        InferenceTile::F32Sm120TmaFmaM128N64
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (12, 0) =>
        {
            "nn_sm120_tma_fma_m128n64_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaM128N64 => return Err(invalid()),
        InferenceTile::F32Sm120TmaFmaM64N128
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (12, 0) =>
        {
            "nn_sm120_tma_fma_m64n128_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaM64N128 => return Err(invalid()),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (12, 0) =>
        {
            "nn_sm120_tma_fma_postbias_m128n64_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64 => return Err(invalid()),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (12, 0) =>
        {
            "nn_sm120_tma_fma_postbias_m64n128_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128 => return Err(invalid()),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (12, 0) =>
        {
            "nn_sm120_tma_fma_postbias_m128n96_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96 => return Err(invalid()),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64K4
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (12, 0) =>
        {
            "nn_sm120_tma_fma_postbias_m128n64_bk16_s2_k4"
        }
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64K4 => return Err(invalid()),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (12, 0) =>
        {
            "nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256 => return Err(invalid()),
        InferenceTile::F32Sm120TmaFmaFixedNoBiasM128N64T256
            if matches!(row, "f32_exact" | "f32_exact_fast") && cc == (12, 0) =>
        {
            "nn_sm120_tma_fma_nobias_m128n64_t256_bk16_s2"
        }
        InferenceTile::F32Sm120TmaFmaFixedNoBiasM128N64T256 => return Err(invalid()),
    };
    let bias_contract = match tile {
        InferenceTile::F32Sm120TmaFmaM128N64
        | InferenceTile::F32Sm120TmaFmaM64N128
        | InferenceTile::F32Sm120TmaFmaFixedNoBiasM128N64T256
        | InferenceTile::TcM64N64Sm89S3
        | InferenceTile::TcM128N64Sm89S2 => FixedForceBiasContract::NoBias,
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64
        | InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128
        | InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96
        | InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64K4
        | InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256 => FixedForceBiasContract::WithBias,
        _ => FixedForceBiasContract::Either,
    };
    Ok(FixedForceSpec {
        tile,
        row,
        input_dtype,
        output_dtype,
        compute_capability: cc,
        bias_contract,
        expected_symbol,
    })
}

fn fixed_explicit_vendor_tiles(row: &str, cc: (u32, u32)) -> Vec<InferenceTile> {
    assert!(matches!(cc, (8, 9) | (12, 0)), "unadmitted comparator CC");
    let mut tiles = match row {
        "bf16" | "f16" => vec![
            InferenceTile::Legacy,
            InferenceTile::Tc16,
            InferenceTile::Tc64,
            InferenceTile::Tc128,
            InferenceTile::TcW64,
            InferenceTile::TcWn64,
        ],
        "bf16_f32" | "f16_f32" => vec![
            InferenceTile::Legacy,
            InferenceTile::Tc16,
            InferenceTile::Tc64,
            InferenceTile::Tc128,
        ],
        "tf32" => vec![
            InferenceTile::Tf32M128S2,
            InferenceTile::Tf32M128S3,
            InferenceTile::Tf32M64S2,
            InferenceTile::Tf32M64S3,
            InferenceTile::Tf32M16S4,
        ],
        "f32_exact" | "f32_exact_fast" => vec![InferenceTile::Legacy, InferenceTile::F32N128S2],
        _ => panic!("unsupported explicit vendor row {row:?}"),
    };
    if cc == (12, 0) {
        match row {
            "bf16" | "f16" | "bf16_f32" | "f16_f32" => {
                tiles.extend(InferenceSm120HalfTile::ALL.map(InferenceTile::Sm120Half))
            }
            "tf32" => tiles.extend([
                InferenceTile::Tf32Sm120M128S2,
                InferenceTile::Tf32Sm120M128S3,
                InferenceTile::Tf32Sm120M64N128S2,
                InferenceTile::Tf32Sm120M64N128S3,
                InferenceTile::Tf32Sm120M64S2ProducerWarp,
                InferenceTile::Tf32Sm120M64S2,
                InferenceTile::Tf32Sm120M64S2PairStore,
            ]),
            "f32_exact" | "f32_exact_fast" => tiles.extend([
                InferenceTile::F32Sm120N64CopyPlan,
                InferenceTile::F32Sm120N64CopyPlanT256,
                InferenceTile::F32Sm120M128N64CopyPlanT256,
                InferenceTile::F32Sm120N64Sliced,
                InferenceTile::F32Sm120TmaFmaM128N64,
                InferenceTile::F32Sm120TmaFmaM64N128,
                InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
                InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128,
                InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96,
                InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64K4,
                InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256,
                InferenceTile::F32Sm120TmaFmaFixedNoBiasM128N64T256,
            ]),
            _ => {}
        }
    } else if row == "tf32" {
        // The wide symbol is not bound in the committed CC12 module set.
        tiles.insert(2, InferenceTile::Tf32M128N128S3);
        tiles.insert(3, InferenceTile::Tf32RnaM128N128S3);
        tiles.insert(4, InferenceTile::Tf32RnaM128N96S3);
    } else if matches!(row, "bf16" | "f16") {
        tiles.push(InferenceTile::Tc128Sm89Pipeline);
        tiles.push(InferenceTile::Tc128Sm89Swizzle);
        tiles.push(InferenceTile::Tc128Sm89S3);
        if row == "f16" {
            tiles.extend([
                InferenceTile::TcM64N64Sm89S3,
                InferenceTile::TcM128N64Sm89S2,
            ]);
        }
    } else if matches!(row, "f32_exact" | "f32_exact_fast") {
        tiles.push(InferenceTile::F32Sm89N64CopyPlan);
    }
    tiles
}

fn fixed_explicit_vendor_filter_tiles(
    inventory: &[InferenceTile],
    requested: Option<&str>,
) -> Result<Vec<InferenceTile>, String> {
    let Some(requested) = requested else {
        return Ok(inventory.to_vec());
    };
    let mut selected = Vec::new();
    for name in requested.split(',') {
        let tile = inventory
            .iter()
            .copied()
            .find(|tile| format!("{tile:?}") == name)
            .ok_or_else(|| {
                format!("unknown MAMBA_FIXED_VENDOR_TILES entry {name:?}; expected {inventory:?}")
            })?;
        if selected.contains(&tile) {
            return Err(format!("duplicate MAMBA_FIXED_VENDOR_TILES entry {name:?}"));
        }
        selected.push(tile);
    }
    Ok(selected)
}

fn fixed_explicit_vendor_paths(requested: Option<&str>) -> Result<Vec<&'static str>, String> {
    let mut paths = Vec::new();
    for name in requested.unwrap_or("eager,graph").split(',') {
        let path = match name {
            "eager" => "eager",
            "graph" => "graph",
            _ => {
                return Err(format!(
                    "unknown MAMBA_FIXED_VENDOR_PATHS entry {name:?}; expected eager,graph"
                ));
            }
        };
        if paths.contains(&path) {
            return Err(format!("duplicate MAMBA_FIXED_VENDOR_PATHS entry {name:?}"));
        }
        paths.push(path);
    }
    Ok(paths)
}

/// One GEMM kernel node as read back from a captured graph: the launch
/// geometry, the driver's parameter layout and the bound pointers.
struct ObservedGemmNode<'a> {
    symbol: &'a str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_bytes: u32,
    driver_abi: Vec<(usize, usize)>,
    terminal_sixth_rejected: bool,
    pointers: [u64; 4],
    bundle: [u32; 8],
}

fn fixed_explicit_vendor_pipeline_graph_contract(
    tile: InferenceTile,
    dtype: WeightDtype,
    node_count: usize,
    node: ObservedGemmNode<'_>,
    expected_pointers: [u64; 4],
    shape: InferenceShape,
) -> Result<(), String> {
    let ObservedGemmNode {
        symbol,
        grid,
        block,
        shared_bytes,
        driver_abi,
        terminal_sixth_rejected,
        pointers,
        bundle,
    } = node;
    if matches!(
        tile,
        InferenceTile::Tf32RnaM128N96S3
            | InferenceTile::TcM64N64Sm89S3
            | InferenceTile::TcM128N64Sm89S2
    ) {
        return fixed_explicit_vendor_finalist_graph_contract(
            tile,
            dtype,
            node_count,
            ObservedGemmNode {
                symbol,
                grid,
                block,
                shared_bytes,
                driver_abi,
                terminal_sixth_rejected,
                pointers,
                bundle,
            },
            expected_pointers,
            shape,
        );
    }
    let suffix = match dtype {
        WeightDtype::Bf16 => "bf16",
        WeightDtype::F16 => "f16",
        WeightDtype::F32 => return Err("Ada half pipeline cannot have F32 storage".into()),
    };
    let (family, expected_shared) = match tile {
        InferenceTile::Tc128Sm89Pipeline => ("pipeline", 71_680),
        InferenceTile::Tc128Sm89Swizzle => ("swizzle", 69_632),
        InferenceTile::Tc128Sm89S3 => ("s3", 98_304),
        _ => return Err(format!("{tile:?} is not an Ada half physical descriptor")),
    };
    let expected = format!("nn_sm89_tc128_{family}_{suffix}");
    if node_count != 1
        || symbol != expected
        || grid
            != (
                (shape.m as u32).div_ceil(128) * (shape.n as u32).div_ceil(128),
                1,
                1,
            )
        || block != (256, 1, 1)
        || shared_bytes != expected_shared
        || driver_abi != [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
        || !terminal_sixth_rejected
        || pointers != expected_pointers
        || bundle
            != [
                1.0f32.to_bits(),
                0.0f32.to_bits(),
                shape.m as u32,
                shape.n as u32,
                shape.k as u32,
                shape.k as u32,
                shape.n as u32,
                shape.n as u32,
            ]
    {
        return Err(format!(
            "wrong physical Ada half pipeline: nodes={node_count} symbol={symbol:?} grid={grid:?} block={block:?} shared={shared_bytes} abi={driver_abi:?} sixth_rejected={terminal_sixth_rejected} pointers={pointers:?} expected_pointers={expected_pointers:?} bundle={bundle:?}"
        ));
    }
    Ok(())
}

fn fixed_explicit_vendor_pair_store_graph_contract(
    node_count: usize,
    symbol: &str,
    block: (u32, u32, u32),
    shared_bytes: u32,
) -> Result<(), String> {
    if node_count != 1
        || symbol != "nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store"
        || block != (128, 1, 1)
        || shared_bytes != 32_896
    {
        return Err(format!(
            "wrong physical SM120 pair-store: nodes={node_count} symbol={symbol:?} block={block:?} shared={shared_bytes}"
        ));
    }
    Ok(())
}

fn fixed_explicit_vendor_needs_identity_graph(paths: &[&str], tile: InferenceTile) -> bool {
    paths.contains(&"graph")
        || fixed_explicit_vendor_ada_descriptor(tile)
        || tile == InferenceTile::Tf32RnaM128N128S3
}

fn fixed_explicit_vendor_ada_descriptor(tile: InferenceTile) -> bool {
    matches!(
        tile,
        InferenceTile::Tc128Sm89Pipeline
            | InferenceTile::Tc128Sm89Swizzle
            | InferenceTile::Tc128Sm89S3
            | InferenceTile::Tf32RnaM128N96S3
            | InferenceTile::TcM64N64Sm89S3
            | InferenceTile::TcM128N64Sm89S2
    )
}

fn fixed_explicit_vendor_finalist_graph_contract(
    tile: InferenceTile,
    dtype: WeightDtype,
    node_count: usize,
    node: ObservedGemmNode<'_>,
    expected_pointers: [u64; 4],
    shape: InferenceShape,
) -> Result<(), String> {
    let ObservedGemmNode {
        symbol,
        grid,
        block,
        shared_bytes,
        driver_abi,
        terminal_sixth_rejected,
        pointers,
        bundle,
    } = node;
    let (row, bm, bn, threads, shared, tf32) = match tile {
        InferenceTile::Tf32RnaM128N96S3 => ("tf32", 128usize, 96usize, 256, 86_016, true),
        InferenceTile::TcM64N64Sm89S3 => ("f16", 64, 64, 128, 49_152, false),
        InferenceTile::TcM128N64Sm89S2 => ("f16", 128, 64, 128, 49_152, false),
        _ => return Err(format!("{tile:?} is not an Ada finalist descriptor")),
    };
    let spec = fixed_force_spec(row, (8, 9), tile)?;
    let legal_shape = if tf32 {
        shape.m > 0
            && shape.k <= i32::MAX as usize - 31
            && shape.k.is_multiple_of(4)
            && shape.n > 0
            && shape.n <= i32::MAX as usize - 95
            && shape.n.is_multiple_of(4)
            && expected_pointers[0] != 0
            && expected_pointers[0].is_multiple_of(4)
            && expected_pointers[3].is_multiple_of(4)
            && (shape.k == 0
                || expected_pointers[1..3]
                    .iter()
                    .all(|pointer| *pointer != 0 && pointer % 16 == 0))
    } else {
        (1..=2048).contains(&shape.m)
            && (shape.k, shape.n) == if bm == 64 { (768, 2304) } else { (2304, 768) }
            && expected_pointers[3] == 0
            && expected_pointers[..3]
                .iter()
                .all(|pointer| *pointer != 0 && pointer % 16 == 0)
    };
    let expected_grid = shape
        .m
        .div_ceil(bm)
        .checked_mul(shape.n.div_ceil(bn))
        .filter(|count| *count <= i32::MAX as usize)
        .and_then(|count| u32::try_from(count).ok())
        .map(|count| (count, 1, 1));
    let (middle, last) = if tf32 {
        (shape.k, shape.n)
    } else {
        (shape.n, shape.k)
    };
    let expected_bundle = [
        1.0f32.to_bits(),
        0,
        shape.m as u32,
        middle as u32,
        last as u32,
        shape.k as u32,
        shape.n as u32,
        shape.n as u32,
    ];
    if !legal_shape
        || shape.m > i32::MAX as usize
        || dtype != spec.input_dtype
        || node_count != 1
        || symbol != spec.expected_symbol
        || Some(grid) != expected_grid
        || block != (threads, 1, 1)
        || shared_bytes != shared
        || driver_abi != [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
        || !terminal_sixth_rejected
        || pointers != expected_pointers
        || bundle != expected_bundle
    {
        return Err(format!(
            "wrong physical Ada finalist {tile:?}: dtype={dtype:?} nodes={node_count} symbol={symbol:?} grid={grid:?} block={block:?} shared={shared_bytes} abi={driver_abi:?} sixth_rejected={terminal_sixth_rejected} pointers={pointers:?} expected_pointers={expected_pointers:?} bundle={bundle:?} shape={shape:?}"
        ));
    }
    Ok(())
}

#[test]
fn fixed_explicit_vendor_finalist_graph_contracts_reject_physical_mutations() {
    for (tile, dtype, shape, grid, block, shared) in [
        (
            InferenceTile::Tf32RnaM128N96S3,
            WeightDtype::F32,
            InferenceShape {
                m: 2048,
                k: 2304,
                n: 768,
            },
            (128, 1, 1),
            (256, 1, 1),
            86_016,
        ),
        (
            InferenceTile::TcM64N64Sm89S3,
            WeightDtype::F16,
            InferenceShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
            (1152, 1, 1),
            (128, 1, 1),
            49_152,
        ),
        (
            InferenceTile::TcM128N64Sm89S2,
            WeightDtype::F16,
            InferenceShape {
                m: 2048,
                k: 2304,
                n: 768,
            },
            (192, 1, 1),
            (128, 1, 1),
            49_152,
        ),
    ] {
        let tf32 = dtype == WeightDtype::F32;
        let row = if tf32 { "tf32" } else { "f16" };
        let symbol = fixed_force_spec(row, (8, 9), tile).unwrap().expected_symbol;
        let abi = vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
        let pointers = [0x1000, 0x2000, 0x3000, 0];
        let bundle = [
            1.0f32.to_bits(),
            0,
            shape.m as u32,
            if tf32 { shape.k } else { shape.n } as u32,
            if tf32 { shape.n } else { shape.k } as u32,
            shape.k as u32,
            shape.n as u32,
            shape.n as u32,
        ];
        let check = |nodes,
                     name,
                     actual_grid,
                     actual_block,
                     actual_shared,
                     actual_abi,
                     terminal,
                     actual_ptrs,
                     actual_bundle| {
            fixed_explicit_vendor_finalist_graph_contract(
                tile,
                dtype,
                nodes,
                ObservedGemmNode {
                    symbol: name,
                    grid: actual_grid,
                    block: actual_block,
                    shared_bytes: actual_shared,
                    driver_abi: actual_abi,
                    terminal_sixth_rejected: terminal,
                    pointers: actual_ptrs,
                    bundle: actual_bundle,
                },
                pointers,
                shape,
            )
        };
        assert!(
            check(
                1,
                symbol,
                grid,
                block,
                shared,
                abi.clone(),
                true,
                pointers,
                bundle
            )
            .is_ok()
        );
        assert!(
            check(
                2,
                symbol,
                grid,
                block,
                shared,
                abi.clone(),
                true,
                pointers,
                bundle
            )
            .is_err()
        );
        assert!(
            check(
                1,
                "wrong",
                grid,
                block,
                shared,
                abi.clone(),
                true,
                pointers,
                bundle
            )
            .is_err()
        );
        assert!(
            check(
                1,
                symbol,
                (grid.0, 2, 1),
                block,
                shared,
                abi.clone(),
                true,
                pointers,
                bundle
            )
            .is_err()
        );
        assert!(
            check(
                1,
                symbol,
                grid,
                (block.0, 2, 1),
                shared,
                abi.clone(),
                true,
                pointers,
                bundle
            )
            .is_err()
        );
        assert!(
            check(
                1,
                symbol,
                grid,
                block,
                shared - 16,
                abi.clone(),
                true,
                pointers,
                bundle
            )
            .is_err()
        );
        assert!(
            check(
                1,
                symbol,
                grid,
                block,
                shared,
                abi.clone(),
                false,
                pointers,
                bundle
            )
            .is_err()
        );
        let mut bad_abi = abi.clone();
        bad_abi[4].1 = 28;
        assert!(
            check(
                1, symbol, grid, block, shared, bad_abi, true, pointers, bundle
            )
            .is_err()
        );
        for index in 0..4 {
            let mut bad = pointers;
            bad[index] ^= 16;
            assert!(
                check(
                    1,
                    symbol,
                    grid,
                    block,
                    shared,
                    abi.clone(),
                    true,
                    bad,
                    bundle
                )
                .is_err()
            );
        }
        for index in 0..8 {
            let mut bad = bundle;
            bad[index] ^= 1;
            assert!(
                check(
                    1,
                    symbol,
                    grid,
                    block,
                    shared,
                    abi.clone(),
                    true,
                    pointers,
                    bad
                )
                .is_err()
            );
        }
    }
}

#[test]
fn fixed_explicit_vendor_rna_wide_eager_needs_identity_graph() {
    assert!(fixed_explicit_vendor_needs_identity_graph(
        &["eager"],
        InferenceTile::Tf32RnaM128N128S3,
    ));
    for tile in [InferenceTile::Tf32RnaM128N128S3, InferenceTile::Tf32M64S2] {
        for paths in [&["graph"][..], &["eager", "graph"][..]] {
            assert!(fixed_explicit_vendor_needs_identity_graph(paths, tile));
        }
    }
    assert!(!fixed_explicit_vendor_needs_identity_graph(
        &["eager"],
        InferenceTile::Tf32M64S2,
    ));
    for tile in [
        InferenceTile::Tc128Sm89Pipeline,
        InferenceTile::Tc128Sm89Swizzle,
        InferenceTile::Tc128Sm89S3,
    ] {
        assert!(
            fixed_explicit_vendor_needs_identity_graph(&["eager"], tile),
            "Ada half {tile:?} needs an untimed identity graph even for eager-only timing"
        );
    }
}

fn fixed_explicit_vendor_rna_wide_graph_contract(
    node_count: usize,
    symbol: &str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_bytes: u32,
    bundle: [u32; 8],
) -> Result<(), String> {
    let m = bundle[2] as i32;
    let k = bundle[3] as i32;
    let n = bundle[4] as i32;
    if node_count != 1
        || symbol != "nn_rna_wide_tf32_m128n128_bk32_s3"
        || block != (256, 1, 1)
        || shared_bytes != 98_304
        || bundle[0] != 1.0f32.to_bits()
        || bundle[1] != 0.0f32.to_bits()
        || m <= 0
        || k < 0
        || n <= 0
        || n > i32::MAX - 127
        || k % 4 != 0
        || n % 4 != 0
        || bundle[5] != bundle[3]
        || bundle[6] != bundle[4]
        || bundle[7] != bundle[4]
    {
        return Err(format!(
            "wrong physical Fixed RNA wide: nodes={node_count} symbol={symbol:?} grid={grid:?} block={block:?} shared={shared_bytes} bundle={bundle:?}"
        ));
    }
    let expected_grid = (m as u32)
        .div_ceil(128)
        .checked_mul((n as u32).div_ceil(128))
        .filter(|count| *count <= i32::MAX as u32)
        .map(|count| (count, 1, 1));
    if Some(grid) != expected_grid {
        return Err(format!(
            "wrong Fixed RNA wide grid {grid:?} for M={m} N={n}"
        ));
    }
    Ok(())
}

#[test]
fn fixed_explicit_vendor_rna_wide_graph_contract_is_exact() {
    let symbol = "nn_rna_wide_tf32_m128n128_bk32_s3";
    let bundle = [1.0f32.to_bits(), 0, 4621, 384, 1928, 384, 1928, 1928];
    let valid = |nodes, name, grid, block, shared, params| {
        fixed_explicit_vendor_rna_wide_graph_contract(nodes, name, grid, block, shared, params)
    };
    assert!(valid(1, symbol, (592, 1, 1), (256, 1, 1), 98_304, bundle).is_ok());
    assert!(
        valid(
            1,
            symbol,
            (2, 1, 1),
            (256, 1, 1),
            98_304,
            [1.0f32.to_bits(), 0, 17, 0, 132, 0, 132, 132],
        )
        .is_ok(),
        "K0 with partial M/N tiles keeps the same launch ABI",
    );
    for (nodes, name, grid, block, shared) in [
        (2, symbol, (592, 1, 1), (256, 1, 1), 98_304),
        (
            1,
            "nn_sm80_mma_tf32_m128n128_bk32_s3",
            (592, 1, 1),
            (256, 1, 1),
            98_304,
        ),
        (1, symbol, (591, 1, 1), (256, 1, 1), 98_304),
        (1, symbol, (592, 2, 1), (256, 1, 1), 98_304),
        (1, symbol, (592, 1, 1), (128, 1, 1), 98_304),
        (1, symbol, (592, 1, 1), (256, 1, 1), 55_296),
    ] {
        assert!(valid(nodes, name, grid, block, shared, bundle).is_err());
    }
    for (index, value) in [
        (0, 0),
        (1, (-0.0f32).to_bits()),
        (2, 0),
        (2, u32::MAX),
        (3, u32::MAX),
        (3, 383),
        (4, 0),
        (4, 1929),
        (4, 2_147_483_644),
        (5, 1928),
        (6, 384),
        (7, 384),
    ] {
        let mut wrong = bundle;
        wrong[index] = value;
        assert!(
            valid(1, symbol, (592, 1, 1), (256, 1, 1), 98_304, wrong).is_err(),
            "wrong captured parameter {index}={value} must be rejected",
        );
    }
}

#[test]
fn fixed_explicit_vendor_pair_store_graph_contract_is_exact() {
    let symbol = "nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store";
    assert!(
        fixed_explicit_vendor_pair_store_graph_contract(1, symbol, (128, 1, 1), 32_896).is_ok()
    );
    for (nodes, name, block, shared) in [
        (2, symbol, (128, 1, 1), 32_896),
        (1, "nn_sm120_tma_tf32_m64n64_bk32_s2", (128, 1, 1), 32_896),
        (1, symbol, (160, 1, 1), 32_896),
        (1, symbol, (128, 1, 1), 32_768),
    ] {
        assert!(
            fixed_explicit_vendor_pair_store_graph_contract(nodes, name, block, shared).is_err()
        );
    }
}

fn fixed_explicit_vendor_raw_bytes(ctx: &GpuCtx, buffer: &DtypedBuf) -> Vec<u8> {
    let mut bytes = vec![0; buffer.size_bytes()];
    if !bytes.is_empty() {
        assert_eq!(
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    bytes.as_mut_ptr().cast(),
                    buffer.cached_ptr(),
                    bytes.len(),
                    ctx.stream.cu_stream(),
                )
            },
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "explicit comparator raw-storage download"
        );
        ctx.stream
            .synchronize()
            .expect("complete raw-storage download");
    }
    bytes
}

fn fixed_explicit_vendor_poison_output(ctx: &GpuCtx, buffer: &DtypedBuf) {
    assert_eq!(
        unsafe {
            cudarc::driver::sys::cuMemsetD8Async(
                buffer.cached_ptr(),
                0xff,
                buffer.size_bytes(),
                ctx.stream.cu_stream(),
            )
        },
        cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "poison explicit comparator output before graph replay"
    );
}

// Inspect the captured Driver graph, not the returned InferenceTile. This runs
// outside capture and timing; the optional bias symbol proves the vendor
// graph includes the broadcast as well as at least one GEMM kernel.
fn fixed_explicit_vendor_graph_inventory(
    graph: &CudaGraph,
    label: &str,
    ada_descriptor: Option<(
        InferenceTile,
        WeightDtype,
        InferenceFwdOperands,
        InferenceShape,
    )>,
    auto_descriptor: Option<(
        FixedAutoPhysicalDescriptor,
        InferenceFwdOperands,
        InferenceShape,
    )>,
    bias_symbol: Option<&str>,
) -> String {
    use cudarc::driver::sys;
    let observed = combined_gemm_acceptance::read_driver_graph(graph)
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    let count = observed.count;
    let mut kernels = Vec::new();
    let non_kernel_nodes = observed.non_kernel_nodes;
    assert!(
        non_kernel_nodes == 0 || (ada_descriptor.is_none() && auto_descriptor.is_none()),
        "{label} pipeline captured non-kernel work"
    );
    let mut bias_count = 0;
    for node in observed.kernels {
        let params = node.params;
        let symbol = node.symbol.as_str();
        let block = (params.blockDimX, params.blockDimY, params.blockDimZ);
        if label == "Tf32RnaM128N128S3" || symbol == "nn_rna_wide_tf32_m128n128_bk32_s3" {
            for (index, expected) in [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
                .into_iter()
                .enumerate()
            {
                let mut offset = 0;
                let mut size = 0;
                assert_eq!(
                    unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) },
                    sys::CUresult::CUDA_SUCCESS,
                    "{label} RNA wide Driver parameter {index}",
                );
                assert_eq!((offset, size), expected, "{label} RNA wide 32-byte ABI");
            }
            let mut offset = 0;
            let mut size = 0;
            assert_eq!(
                unsafe { sys::cuFuncGetParamInfo(params.func, 5, &mut offset, &mut size) },
                sys::CUresult::CUDA_ERROR_INVALID_VALUE,
                "{label} RNA wide must not have a sixth argument",
            );
            assert!(!params.kernelParams.is_null());
            let bundle_pointer = unsafe { *params.kernelParams.add(4) };
            assert!(!bundle_pointer.is_null());
            let bundle = unsafe { bundle_pointer.cast::<[u32; 8]>().read_unaligned() };
            fixed_explicit_vendor_rna_wide_graph_contract(
                count,
                symbol,
                (params.gridDimX, params.gridDimY, params.gridDimZ),
                block,
                params.sharedMemBytes,
                bundle,
            )
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        }
        if label == "Tf32Sm120M64S2PairStore" {
            fixed_explicit_vendor_pair_store_graph_contract(
                count,
                symbol,
                block,
                params.sharedMemBytes,
            )
            .unwrap_or_else(|error| panic!("{label}: {error}"));
            // CUDA 12 uses 64-byte map alignment; CUDA 13 uses 128.
            // The Fixed loader already checks host/compiler agreement.
            let map_offset = std::mem::align_of::<sys::CUtensorMap>();
            for (index, expected) in [
                (0, (0, 8)),
                (1, (map_offset, 128)),
                (2, (map_offset + 128, 128)),
                (3, (map_offset + 256, 8)),
                (4, (map_offset + 264, 16)),
            ] {
                let mut offset = 0;
                let mut size = 0;
                assert_eq!(
                    unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) },
                    sys::CUresult::CUDA_SUCCESS
                );
                assert_eq!((offset, size), expected, "{label} five-argument TMA ABI");
            }
            let mut offset = 0;
            let mut size = 0;
            assert_eq!(
                unsafe { sys::cuFuncGetParamInfo(params.func, 5, &mut offset, &mut size) },
                sys::CUresult::CUDA_ERROR_INVALID_VALUE
            );
            assert!(!params.kernelParams.is_null());
            let dimensions = unsafe { *(*params.kernelParams.add(4)).cast::<[i32; 4]>() };
            assert!(dimensions[0] > 0 && dimensions[2] > 0);
            assert_eq!(dimensions[3], dimensions[2]);
            assert_eq!(
                (params.gridDimX, params.gridDimY, params.gridDimZ),
                (
                    (dimensions[0] as u32).div_ceil(64) * (dimensions[2] as u32).div_ceil(64),
                    1,
                    1
                )
            );
        }
        if let Some((descriptor, operands, shape)) = auto_descriptor {
            let mut driver_abi = Vec::with_capacity(5);
            for index in 0..5 {
                let mut offset = 0;
                let mut size = 0;
                assert_eq!(
                    unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) },
                    sys::CUresult::CUDA_SUCCESS,
                    "{label} retained AUTO Driver parameter {index}",
                );
                driver_abi.push((offset, size));
            }
            let mut offset = 0;
            let mut size = 0;
            let terminal_sixth_rejected =
                unsafe { sys::cuFuncGetParamInfo(params.func, 5, &mut offset, &mut size) }
                    == sys::CUresult::CUDA_ERROR_INVALID_VALUE;
            let mut static_shared_bytes = 0;
            assert_eq!(
                unsafe {
                    sys::cuFuncGetAttribute(
                        &mut static_shared_bytes,
                        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                        params.func,
                    )
                },
                sys::CUresult::CUDA_SUCCESS,
                "{label} retained AUTO static shared memory",
            );
            let static_shared_bytes = u32::try_from(static_shared_bytes)
                .unwrap_or_else(|_| panic!("{label} retained AUTO static shared is negative"));
            assert!(
                !params.kernelParams.is_null(),
                "{label} retained AUTO kernelParams"
            );
            let mut pointers = [0; 4];
            for (index, pointer) in pointers.iter_mut().enumerate() {
                let argument = unsafe { *params.kernelParams.add(index) };
                assert!(
                    !argument.is_null(),
                    "{label} retained AUTO argument {index}"
                );
                *pointer = unsafe { argument.cast::<u64>().read_unaligned() };
            }
            let bundle_pointer = unsafe { *params.kernelParams.add(4) };
            assert!(
                !bundle_pointer.is_null(),
                "{label} retained AUTO parameter bundle"
            );
            let bundle = unsafe { bundle_pointer.cast::<[u32; 8]>().read_unaligned() };
            fixed_auto_bundle_graph_contract(
                descriptor,
                &FixedAutoGraphObservation {
                    node_count: count,
                    symbol,
                    grid: (params.gridDimX, params.gridDimY, params.gridDimZ),
                    block,
                    dynamic_shared_bytes: params.sharedMemBytes,
                    static_shared_bytes,
                    driver_abi,
                    terminal_sixth_rejected,
                    pointers,
                    bundle,
                },
                [
                    operands.c.ptr,
                    operands.x.ptr,
                    operands.w.ptr,
                    operands.bias_ptr.unwrap_or(0),
                ],
                shape,
            )
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        }
        if let Some((tile, dtype, operands, shape)) = ada_descriptor {
            let mut driver_abi = Vec::with_capacity(5);
            for index in 0..5 {
                let mut offset = 0;
                let mut size = 0;
                assert_eq!(
                    unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) },
                    sys::CUresult::CUDA_SUCCESS,
                    "{label} Ada half Driver parameter {index}",
                );
                driver_abi.push((offset, size));
            }
            let mut offset = 0;
            let mut size = 0;
            let terminal_sixth_rejected =
                unsafe { sys::cuFuncGetParamInfo(params.func, 5, &mut offset, &mut size) }
                    == sys::CUresult::CUDA_ERROR_INVALID_VALUE;
            assert!(
                !params.kernelParams.is_null(),
                "{label} Ada half kernelParams"
            );
            let mut pointers = [0; 4];
            for (index, pointer) in pointers.iter_mut().enumerate() {
                let argument = unsafe { *params.kernelParams.add(index) };
                assert!(!argument.is_null(), "{label} Ada half argument {index}");
                *pointer = unsafe { argument.cast::<u64>().read_unaligned() };
            }
            let bundle_pointer = unsafe { *params.kernelParams.add(4) };
            assert!(!bundle_pointer.is_null(), "{label} Ada half bundle");
            let bundle = unsafe { bundle_pointer.cast::<[u32; 8]>().read_unaligned() };
            fixed_explicit_vendor_pipeline_graph_contract(
                tile,
                dtype,
                count,
                ObservedGemmNode {
                    symbol,
                    grid: (params.gridDimX, params.gridDimY, params.gridDimZ),
                    block,
                    shared_bytes: params.sharedMemBytes,
                    driver_abi,
                    terminal_sixth_rejected,
                    pointers,
                    bundle,
                },
                [
                    operands.c.ptr,
                    operands.x.ptr,
                    operands.w.ptr,
                    operands.bias_ptr.unwrap_or(0),
                ],
                shape,
            )
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        }
        bias_count += usize::from(bias_symbol == Some(symbol));
        kernels.push(format!(
            "{{\"symbol\":\"{}\",\"grid\":[{},{},{}],\"block\":[{},{},{}],\"shared_bytes\":{}}}",
            fixed_sm120_tf32_bd_json_escape(symbol),
            params.gridDimX,
            params.gridDimY,
            params.gridDimZ,
            block.0,
            block.1,
            block.2,
            params.sharedMemBytes
        ));
    }
    assert!(!kernels.is_empty(), "{label} graph has no actual kernels");
    if bias_symbol.is_some() {
        assert_eq!(
            bias_count, 1,
            "{label} must capture exactly one bias broadcast"
        );
        assert!(
            kernels.len() >= 2,
            "{label} captured bias but no GEMM kernel"
        );
    }
    format!(
        "{{\"node_count\":{count},\"non_kernel_nodes\":{non_kernel_nodes},\"kernels\":[{}]}}",
        kernels.join(",")
    )
}

#[test]
fn fixed_explicit_vendor_cc_admission_is_fail_closed() {
    assert_eq!(fixed_explicit_vendor_admit_cc(None, (8, 9)), Ok((8, 9)));
    assert_eq!(
        fixed_explicit_vendor_admit_cc(Some("8.9"), (8, 9)),
        Ok((8, 9))
    );
    assert_eq!(
        fixed_explicit_vendor_admit_cc(Some("12.0"), (12, 0)),
        Ok((12, 0))
    );
    assert!(fixed_explicit_vendor_admit_cc(None, (12, 0)).is_err());
    assert!(fixed_explicit_vendor_admit_cc(Some("8.9"), (12, 0)).is_err());
    assert!(fixed_explicit_vendor_admit_cc(Some("12.0"), (8, 9)).is_err());
    assert!(fixed_explicit_vendor_admit_cc(Some("12.0"), (12, 1)).is_err());
    for invalid in ["", "12.1", "9.0", "120", " 12.0", "12.0 ", "8.9,12.0"] {
        assert!(fixed_explicit_vendor_admit_cc(Some(invalid), (12, 0)).is_err());
    }
}

#[test]
fn fixed_explicit_vendor_rna_wide_force_filter_is_ada_tf32_only() {
    let name = "Tf32RnaM128N128S3";
    let selected = fixed_explicit_vendor_filter_tiles(
        &fixed_explicit_vendor_tiles("tf32", (8, 9)),
        Some(name),
    )
    .expect("RNA-compatible wide must be reachable through the Ada TF32 inventory");
    assert_eq!(selected.len(), 1);
    assert_eq!(format!("{:?}", selected[0]), name);
    assert_eq!(
        fixed_force_spec("tf32", (8, 9), selected[0])
            .expect("RNA wide physical force identity")
            .expected_symbol,
        "nn_rna_wide_tf32_m128n128_bk32_s3",
    );
    for (row, cc) in [("tf32", (12, 0)), ("f32_exact", (8, 9)), ("bf16", (8, 9))] {
        assert!(
            fixed_explicit_vendor_filter_tiles(&fixed_explicit_vendor_tiles(row, cc), Some(name))
                .is_err(),
            "RNA wide must not leak into {row}/CC{cc:?}",
        );
    }
}

#[test]
fn fixed_explicit_vendor_pair_store_force_filter_is_sm120_only() {
    let name = "Tf32Sm120M64S2PairStore";
    let selected = fixed_explicit_vendor_filter_tiles(
        &fixed_explicit_vendor_tiles("tf32", (12, 0)),
        Some(name),
    )
    .expect("SM120 pair-store must have an explicit force-only route");
    assert_eq!(selected.len(), 1);
    assert_eq!(format!("{:?}", selected[0]), name);
    for (row, cc) in [("tf32", (8, 9)), ("f32_exact", (12, 0)), ("bf16", (12, 0))] {
        assert!(
            fixed_explicit_vendor_filter_tiles(&fixed_explicit_vendor_tiles(row, cc), Some(name),)
                .is_err()
        );
    }
}

#[test]
fn fixed_explicit_vendor_rung_inventory_is_arch_specific() {
    for row in ["bf16", "f16"] {
        let sm120 = fixed_explicit_vendor_tiles(row, (12, 0));
        assert_eq!(sm120.len(), 11);
        assert!(sm120.contains(&InferenceTile::Legacy));
        for tile in InferenceSm120HalfTile::ALL {
            assert!(sm120.contains(&InferenceTile::Sm120Half(tile)));
        }
        assert!(!sm120.contains(&InferenceTile::Tc128Sm89Pipeline));
        assert!(!sm120.contains(&InferenceTile::Tc128Sm89Swizzle));
        assert!(!sm120.contains(&InferenceTile::Tc128Sm89S3));
        let ada = fixed_explicit_vendor_tiles(row, (8, 9));
        assert_eq!(ada.len(), if row == "f16" { 11 } else { 9 });
        assert!(ada.contains(&InferenceTile::Legacy));
        assert!(ada.contains(&InferenceTile::Tc128Sm89Pipeline));
        assert!(ada.contains(&InferenceTile::Tc128Sm89Swizzle));
        assert!(ada.contains(&InferenceTile::Tc128Sm89S3));
        for tile in [
            InferenceTile::TcM64N64Sm89S3,
            InferenceTile::TcM128N64Sm89S2,
        ] {
            assert_eq!(ada.contains(&tile), row == "f16");
            assert!(!sm120.contains(&tile));
        }
        assert!(
            ada.iter()
                .all(|tile| !matches!(tile, InferenceTile::Sm120Half(_)))
        );
    }
    let sm120 = fixed_explicit_vendor_tiles("tf32", (12, 0));
    assert_eq!(sm120.len(), 12);
    assert!(!sm120.contains(&InferenceTile::Tf32M128N128S3));
    assert!(!sm120.contains(&InferenceTile::Tf32RnaM128N128S3));
    for tile in [
        InferenceTile::Tf32Sm120M128S2,
        InferenceTile::Tf32Sm120M128S3,
        InferenceTile::Tf32Sm120M64N128S2,
        InferenceTile::Tf32Sm120M64N128S3,
        InferenceTile::Tf32Sm120M64S2ProducerWarp,
        InferenceTile::Tf32Sm120M64S2,
        InferenceTile::Tf32Sm120M64S2PairStore,
    ] {
        assert!(sm120.contains(&tile));
    }
    let ada = fixed_explicit_vendor_tiles("tf32", (8, 9));
    assert_eq!(ada.len(), 8);
    assert!(ada.contains(&InferenceTile::Tf32M128N128S3));
    assert!(ada.contains(&InferenceTile::Tf32RnaM128N128S3));
    assert!(ada.contains(&InferenceTile::Tf32RnaM128N96S3));
    assert!(!sm120.contains(&InferenceTile::Tf32RnaM128N96S3));
    assert!(!ada.contains(&InferenceTile::Tc128Sm89Pipeline));
    assert_eq!(
        fixed_explicit_vendor_tiles("f32_exact", (8, 9)),
        [
            InferenceTile::Legacy,
            InferenceTile::F32N128S2,
            InferenceTile::F32Sm89N64CopyPlan,
        ]
    );
    assert_eq!(
        fixed_explicit_vendor_tiles("f32_exact", (12, 0)),
        [
            InferenceTile::Legacy,
            InferenceTile::F32N128S2,
            InferenceTile::F32Sm120N64CopyPlan,
            InferenceTile::F32Sm120N64CopyPlanT256,
            InferenceTile::F32Sm120M128N64CopyPlanT256,
            InferenceTile::F32Sm120N64Sliced,
            InferenceTile::F32Sm120TmaFmaM128N64,
            InferenceTile::F32Sm120TmaFmaM64N128,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM64N128,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64K4,
            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256,
            InferenceTile::F32Sm120TmaFmaFixedNoBiasM128N64T256,
        ]
    );
    for row in ["bf16_f32", "f16_f32"] {
        assert_eq!(fixed_explicit_vendor_tiles(row, (8, 9)).len(), 4);
        let sm120 = fixed_explicit_vendor_tiles(row, (12, 0));
        assert_eq!(sm120.len(), 9);
        assert!(sm120.contains(&InferenceTile::Legacy));
        for tile in InferenceSm120HalfTile::ALL {
            assert!(sm120.contains(&InferenceTile::Sm120Half(tile)));
        }
    }
}

#[test]
fn fixed_explicit_vendor_exact_rows_keep_distinct_denominators_and_tolerances() {
    use cudarc::cublas::sys::cublasComputeType_t;

    let rows = fixed_explicit_vendor_row_specs();
    let pedantic = rows
        .iter()
        .find(|row| row.name == "f32_exact")
        .expect("PEDANTIC exact comparator row");
    let fast = rows
        .iter()
        .find(|row| row.name == "f32_exact_fast")
        .expect("FAST_TF32 exact comparator row");

    for row in [pedantic, fast] {
        assert_eq!(row.input_dtype, WeightDtype::F32);
        assert_eq!(row.output_dtype, WeightDtype::F32);
        assert_eq!(row.policy, F32TriadPolicy::ExactScalarFma);
        assert_eq!(row.custom_tolerance, 0.0002);
    }
    assert_eq!(
        pedantic.vendor_compute,
        cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC
    );
    assert_eq!(pedantic.vendor_tolerance, 0.0002);
    assert_eq!(pedantic.vendor_comparator, "CUBLAS_COMPUTE_32F_PEDANTIC");
    assert_eq!(
        fast.vendor_compute,
        cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
    );
    assert_eq!(fast.vendor_tolerance, 0.0025);
    assert_eq!(fast.vendor_comparator, "CUBLAS_COMPUTE_32F_FAST_TF32");
    assert_eq!(
        fixed_exact_comparator_completion_metadata(),
        "\"exact_comparators\":{\"f32_exact\":\"CUBLAS_COMPUTE_32F_PEDANTIC\",\"f32_exact_fast\":\"CUBLAS_COMPUTE_32F_FAST_TF32\"}"
    );
    assert_eq!(
        fixed_explicit_vendor_tolerance_metadata(fast),
        "\"vendor_comparator\":\"CUBLAS_COMPUTE_32F_FAST_TF32\",\"normalized_error_tolerance\":0.0002,\"custom_normalized_error_tolerance\":0.0002,\"vendor_normalized_error_tolerance\":0.0025"
    );
}

#[test]
fn fixed_production_auto_adapter_inventory_and_record_count_are_exact() {
    let rows = fixed_explicit_vendor_row_specs();
    assert_eq!(
        rows.map(|row| (row.name, row.input_dtype, row.output_dtype)),
        [
            ("bf16", WeightDtype::Bf16, WeightDtype::Bf16),
            ("f16", WeightDtype::F16, WeightDtype::F16),
            ("bf16_f32", WeightDtype::Bf16, WeightDtype::F32),
            ("f16_f32", WeightDtype::F16, WeightDtype::F32),
            ("tf32", WeightDtype::F32, WeightDtype::F32),
            ("f32_exact", WeightDtype::F32, WeightDtype::F32),
            ("f32_exact_fast", WeightDtype::F32, WeightDtype::F32),
        ]
    );
    assert_eq!(FIXED_PRODUCTION_AUTO_DEFAULT_WINDOWS, 21);
    assert_eq!(fixed_production_auto_expected_records(7, 5, 2, 2), 280);
}

#[test]
fn fixed_production_auto_ratio_orientation_is_auto_over_vendor() {
    assert_eq!(
        fixed_production_auto_ratio_samples(&[4.0, 12.0], &[8.0, 6.0]).unwrap(),
        [0.5, 2.0]
    );
    assert!(fixed_production_auto_ratio_samples(&[1.0], &[1.0, 2.0]).is_err());
    assert!(fixed_production_auto_ratio_samples(&[1.0], &[0.0]).is_err());
}

#[test]
fn fixed_production_auto_graph_symbol_membership_is_exact() {
    let base = "nn_sm120_tma_tf32_m64n64_bk32_s2";
    let producer = "nn_sm120_tma_tf32_m64n64_bk32_s2_producer_warp";
    let pair_store = "nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store";
    let inventory =
        format!("{{\"kernels\":[{{\"symbol\":\"{producer}\"}},{{\"symbol\":\"{pair_store}\"}}]}}");
    assert!(!fixed_production_auto_inventory_has_exact_symbol(
        &inventory, base
    ));
    assert!(fixed_production_auto_inventory_has_exact_symbol(
        &inventory, producer
    ));
    assert!(fixed_production_auto_inventory_has_exact_symbol(
        &inventory, pair_store
    ));
}

#[test]
fn fixed_production_auto_cuda_uuid_and_tuning_revision_are_pinned() {
    let bytes = [
        0x12,
        0x34,
        0x56,
        0x78,
        0x9a_u8 as i8,
        0xbc_u8 as i8,
        0xde_u8 as i8,
        0xf0_u8 as i8,
        0x10,
        0x20,
        0x30,
        0x40,
        0x50,
        0x60,
        0x70,
        0x80_u8 as i8,
    ];
    assert_eq!(
        fixed_production_auto_format_cuda_uuid(bytes),
        "GPU-12345678-9abc-def0-1020-304050607080"
    );
    assert_eq!(
        fixed_production_auto_tuning_metadata(),
        "\"tuning_table_revision\":45"
    );
}

#[test]
fn fixed_explicit_vendor_fast_exact_row_reuses_every_exact_force_spec_once() {
    for cc in [(8, 9), (12, 0)] {
        let pedantic = fixed_explicit_vendor_tiles("f32_exact", cc);
        let fast = fixed_explicit_vendor_tiles("f32_exact_fast", cc);
        assert_eq!(fast, pedantic);
        assert_eq!(fast.len(), if cc == (8, 9) { 3 } else { 14 });
        for tile in fast {
            let fast_spec = fixed_force_spec("f32_exact_fast", cc, tile)
                .expect("FAST_TF32 denominator must retain exact custom force spec");
            let pedantic_spec =
                fixed_force_spec("f32_exact", cc, tile).expect("PEDANTIC exact custom force spec");
            assert_eq!(fast_spec.input_dtype, pedantic_spec.input_dtype);
            assert_eq!(fast_spec.output_dtype, pedantic_spec.output_dtype);
            assert_eq!(fast_spec.bias_contract, pedantic_spec.bias_contract);
            assert_eq!(fast_spec.expected_symbol, pedantic_spec.expected_symbol);
        }
    }
}

#[test]
fn fixed_force_specs_bind_mixed_and_exact_tiles_to_physical_symbols() {
    let mixed = fixed_force_spec(
        "bf16_f32",
        (12, 0),
        InferenceTile::Sm120Half(InferenceSm120HalfTile::M128N64Bk32S3),
    )
    .expect("SM120 mixed-output tile must have a force spec");
    assert_eq!(mixed.input_dtype, WeightDtype::Bf16);
    assert_eq!(mixed.output_dtype, WeightDtype::F32);
    assert_eq!(
        mixed.expected_symbol,
        "nn_sm120_tma_128x64_bk32_s3_f32out_bf16"
    );
    assert!(mixed.bias_contract.allows(false));
    assert!(mixed.bias_contract.allows(true));

    let postbias = fixed_force_spec(
        "f32_exact",
        (12, 0),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64K4,
    )
    .expect("SM120 post-bias tile must have a force spec");
    assert_eq!(
        postbias.expected_symbol,
        "nn_sm120_tma_fma_postbias_m128n64_bk16_s2_k4"
    );
    assert!(!postbias.bias_contract.allows(false));
    assert!(postbias.bias_contract.allows(true));

    let nobias = fixed_force_spec(
        "f32_exact",
        (12, 0),
        InferenceTile::F32Sm120TmaFmaFixedNoBiasM128N64T256,
    )
    .expect("SM120 no-bias tile must have a force spec");
    assert!(nobias.bias_contract.allows(false));
    assert!(!nobias.bias_contract.allows(true));
}

#[cfg(test)]
fn inference_bundle_auto_tooling_request(
    nvrtc: (i32, i32),
    state_capacity: usize,
    row: &'static str,
    input_dtype: WeightDtype,
    shape: InferenceShape,
    bias_ptr: Option<u64>,
) -> FixedAutoPhysicalRequest<'static> {
    FixedAutoPhysicalRequest {
        row,
        operands: InferenceFwdOperands {
            c: TypedPtr {
                ptr: 0x1000,
                dtype: WeightDtype::F32,
            },
            x: TypedPtr {
                ptr: 0x2000,
                dtype: input_dtype,
            },
            w: TypedPtr {
                ptr: 0x3000,
                dtype: input_dtype,
            },
            bias_ptr,
        },
        shape,
        selected: if input_dtype == WeightDtype::F32 {
            InferenceTile::F32Sm89N64CopyPlan
        } else {
            InferenceTile::Tc128Sm89S3
        },
        compute_capability: (8, 9),
        multiprocessors: 142,
        compiler_target: "sm_89",
        state_capacity,
        nvrtc,
        nvrtc_library_known: true,
        policy: F32TriadPolicy::ExactScalarFma,
    }
}

#[cfg(test)]
fn inference_bundle_auto_tooling_recorded_route(
    symbol: &'static str,
    dtype: mamba_rs::mamba_ssm::gpu::kernel_identity::PolicyDtype,
    backend: mamba_rs::mamba_ssm::gpu::kernel_identity::PhysicalGemmBackend,
    shape: InferenceShape,
    geometry: ((u32, u32), u32, u8),
    launch: ((u32, u32, u32), (u32, u32, u32), u32),
) -> FixedAutoRecordedRoute {
    FixedAutoRecordedRoute {
        op: ResolvedGemmOp::Nn,
        symbol,
        dtype,
        backend,
        shape: (shape.m, shape.k, shape.n),
        strides: (shape.k, shape.n, shape.n),
        tile: geometry.0,
        bk: geometry.1,
        stages: geometry.2,
        threads: launch.1.0,
        grid: launch.0,
        block: launch.1,
        dynamic_shared_bytes: launch.2,
    }
}

#[test]
fn inference_bundle_auto_tooling_resolves_literal_cohorts_without_rewriting_force_aliases() {
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{PhysicalGemmBackend, PolicyDtype};

    let hot_b = InferenceShape {
        m: 4621,
        k: 768,
        n: 2304,
    };
    let hot_c = InferenceShape {
        m: 4621,
        k: 1928,
        n: 384,
    };
    let abi = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
    let cohorts = [
        ((12, 8), 16),
        ((12, 8), 64),
        ((13, 0), 16),
        ((13, 0), 64),
        ((13, 2), 16),
        ((13, 2), 64),
    ];
    assert_eq!(cohorts.len(), 6);
    for (nvrtc, state_capacity) in cohorts {
        for (row, dtype, policy_dtype, symbol) in [
            (
                "bf16_f32",
                WeightDtype::Bf16,
                PolicyDtype::Bf16,
                "nn_sm89_tc128_f32out_s3_bf16",
            ),
            (
                "f16_f32",
                WeightDtype::F16,
                PolicyDtype::F16,
                "nn_sm89_tc128_f32out_s3_f16",
            ),
        ] {
            for (shape, grid) in [(hot_b, 666), (hot_c, 111)] {
                for bias_ptr in [None, Some(0x4004)] {
                    let request = inference_bundle_auto_tooling_request(
                        nvrtc,
                        state_capacity,
                        row,
                        dtype,
                        shape,
                        bias_ptr,
                    );
                    let route = inference_bundle_auto_tooling_recorded_route(
                        symbol,
                        policy_dtype,
                        PhysicalGemmBackend::InferenceMma16,
                        shape,
                        ((128, 128), 64, 3),
                        ((grid, 1, 1), (256, 1, 1), 98_304),
                    );
                    let descriptor = fixed_auto_bundle_physical_descriptor(request, &[route])
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        descriptor,
                        FixedAutoPhysicalDescriptor {
                            family: InferenceTile::Tc128Sm89S3,
                            symbol,
                            storage: [dtype, dtype, WeightDtype::F32],
                            backend: PhysicalGemmBackend::InferenceMma16,
                            tile: (128, 128),
                            bk: 64,
                            stages: 3,
                            grid: (grid, 1, 1),
                            block: (256, 1, 1),
                            dynamic_shared_bytes: 98_304,
                            static_shared_bytes: 0,
                            driver_abi: abi,
                        },
                        "literal mixed AUTO descriptor for {nvrtc:?}/cap{state_capacity} {row} {shape:?} bias={bias_ptr:?}"
                    );
                    let bundle = if shape == hot_b {
                        [1.0f32.to_bits(), 0, 4621, 2304, 768, 768, 2304, 2304]
                    } else {
                        [1.0f32.to_bits(), 0, 4621, 384, 1928, 1928, 384, 384]
                    };
                    let pointers = [0x1000, 0x2000, 0x3000, bias_ptr.unwrap_or(0)];
                    assert!(
                        fixed_auto_bundle_graph_contract(
                            descriptor,
                            &FixedAutoGraphObservation {
                                node_count: 1,
                                symbol,
                                grid: (grid, 1, 1),
                                block: (256, 1, 1),
                                dynamic_shared_bytes: 98_304,
                                static_shared_bytes: 0,
                                driver_abi: abi.to_vec(),
                                terminal_sixth_rejected: true,
                                pointers,
                                bundle,
                            },
                            pointers,
                            shape,
                        )
                        .is_ok()
                    );
                }
            }
        }

        for row in ["f32_exact", "f32_exact_fast"] {
            let request = inference_bundle_auto_tooling_request(
                nvrtc,
                state_capacity,
                row,
                WeightDtype::F32,
                hot_c,
                None,
            );
            let symbol = "nn_sm89_f32_m128n64_tail_copyplan";
            let route = inference_bundle_auto_tooling_recorded_route(
                symbol,
                PolicyDtype::F32,
                PhysicalGemmBackend::InferenceScalarFma,
                hot_c,
                ((128, 64), 32, 2),
                ((222, 1, 1), (256, 1, 1), 0),
            );
            assert_eq!(
                fixed_auto_bundle_physical_descriptor(request, &[route])
                    .unwrap()
                    .unwrap(),
                FixedAutoPhysicalDescriptor {
                    family: InferenceTile::F32Sm89N64CopyPlan,
                    symbol,
                    storage: [WeightDtype::F32; 3],
                    backend: PhysicalGemmBackend::InferenceScalarFma,
                    tile: (128, 64),
                    bk: 32,
                    stages: 2,
                    grid: (222, 1, 1),
                    block: (256, 1, 1),
                    dynamic_shared_bytes: 0,
                    static_shared_bytes: 49_152,
                    driver_abi: abi,
                },
                "literal exact AUTO descriptor for {nvrtc:?}/cap{state_capacity} {row}"
            );
        }
    }

    assert_eq!(
        fixed_force_spec("bf16", (8, 9), InferenceTile::Tc128Sm89S3)
            .unwrap()
            .expected_symbol,
        "nn_sm89_tc128_s3_bf16"
    );
    assert!(fixed_force_spec("bf16_f32", (8, 9), InferenceTile::Tc128Sm89S3).is_err());
    assert_eq!(
        fixed_force_spec("f32_exact", (8, 9), InferenceTile::F32Sm89N64CopyPlan,)
            .unwrap()
            .expected_symbol,
        "nn_sm89_f32_n64_copyplan"
    );
}

#[test]
fn inference_bundle_auto_tooling_rejects_request_route_and_graph_mutations_closed() {
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{PhysicalGemmBackend, PolicyDtype};

    let shape = InferenceShape {
        m: 4621,
        k: 1928,
        n: 384,
    };
    let request = inference_bundle_auto_tooling_request(
        (13, 2),
        64,
        "f32_exact",
        WeightDtype::F32,
        shape,
        None,
    );
    let symbol = "nn_sm89_f32_m128n64_tail_copyplan";
    let route = inference_bundle_auto_tooling_recorded_route(
        symbol,
        PolicyDtype::F32,
        PhysicalGemmBackend::InferenceScalarFma,
        shape,
        ((128, 64), 32, 2),
        ((222, 1, 1), (256, 1, 1), 0),
    );
    let descriptor = fixed_auto_bundle_physical_descriptor(request, &[route])
        .unwrap()
        .unwrap();

    let mut fallback_request = request;
    fallback_request.selected = InferenceTile::Legacy;
    let error = fixed_auto_bundle_physical_descriptor(fallback_request, &[route]).unwrap_err();
    assert!(error.contains(symbol));
    assert!(error.contains("holder"));

    for wrong_symbol in [
        "nn_sm89_f32_n64_copyplan",
        "nn_sm89_f32_m128n64_tail_copyplan_extra",
        "prefix_gemm_bi_nn_inference_sm89_f32_m128n64_tail_copyplan",
    ] {
        let mut changed = route;
        changed.symbol = wrong_symbol;
        assert!(fixed_auto_bundle_physical_descriptor(request, &[changed]).is_err());
    }
    for changed in [
        FixedAutoRecordedRoute {
            grid: (438, 1, 1),
            ..route
        },
        FixedAutoRecordedRoute {
            block: (128, 1, 1),
            threads: 128,
            ..route
        },
        FixedAutoRecordedRoute {
            dynamic_shared_bytes: 4,
            ..route
        },
        FixedAutoRecordedRoute {
            tile: (64, 64),
            ..route
        },
        FixedAutoRecordedRoute {
            shape: (4620, 1928, 384),
            ..route
        },
        FixedAutoRecordedRoute {
            strides: (1928, 384, 383),
            ..route
        },
    ] {
        assert!(fixed_auto_bundle_physical_descriptor(request, &[changed]).is_err());
    }
    assert!(fixed_auto_bundle_physical_descriptor(request, &[]).is_err());
    assert!(fixed_auto_bundle_physical_descriptor(request, &[route, route]).is_err());

    let mut request_mutations = Vec::new();
    let mut changed = request;
    changed.compute_capability = (8, 8);
    request_mutations.push(changed);
    let mut changed = request;
    changed.multiprocessors = 141;
    request_mutations.push(changed);
    let mut changed = request;
    changed.compiler_target = "compute_89";
    request_mutations.push(changed);
    let mut changed = request;
    changed.state_capacity = 32;
    request_mutations.push(changed);
    let mut changed = request;
    changed.nvrtc = (13, 1);
    request_mutations.push(changed);
    let mut changed = request;
    changed.nvrtc_library_known = false;
    request_mutations.push(changed);
    let mut changed = request;
    changed.policy = F32TriadPolicy::AllowDeterministicTf32;
    request_mutations.push(changed);
    let mut changed = request;
    changed.operands.bias_ptr = Some(0x4004);
    request_mutations.push(changed);
    let mut changed = request;
    changed.shape.m -= 1;
    request_mutations.push(changed);
    let mut changed = request;
    changed.operands.x.ptr = 0x2008;
    request_mutations.push(changed);
    let mut changed = request;
    changed.operands.w.dtype = WeightDtype::F16;
    request_mutations.push(changed);
    for changed in request_mutations {
        assert!(
            fixed_auto_bundle_physical_descriptor(changed, &[route])
                .unwrap()
                .is_none(),
            "closed request mutation entered the retained AUTO domain"
        );
    }

    let pointers = [0x1000, 0x2000, 0x3000, 0];
    let bundle = [1.0f32.to_bits(), 0, 4621, 384, 1928, 1928, 384, 384];
    let good_graph = FixedAutoGraphObservation {
        node_count: 1,
        symbol,
        grid: (222, 1, 1),
        block: (256, 1, 1),
        dynamic_shared_bytes: 0,
        static_shared_bytes: 49_152,
        driver_abi: vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)],
        terminal_sixth_rejected: true,
        pointers,
        bundle,
    };
    assert!(fixed_auto_bundle_graph_contract(descriptor, &good_graph, pointers, shape).is_ok());
    let mut graph_mutations = Vec::new();
    let mut changed = good_graph.clone();
    changed.symbol = "nn_sm89_f32_m128n64_tail_copyplan_extra";
    graph_mutations.push(changed);
    let mut changed = good_graph.clone();
    changed.grid = (438, 1, 1);
    graph_mutations.push(changed);
    let mut changed = good_graph.clone();
    changed.block = (128, 1, 1);
    graph_mutations.push(changed);
    let mut changed = good_graph.clone();
    changed.dynamic_shared_bytes = 4;
    graph_mutations.push(changed);
    let mut changed = good_graph.clone();
    changed.static_shared_bytes = 49_148;
    graph_mutations.push(changed);
    let mut changed = good_graph.clone();
    changed.driver_abi[4] = (32, 28);
    graph_mutations.push(changed);
    let mut changed = good_graph.clone();
    changed.terminal_sixth_rejected = false;
    graph_mutations.push(changed);
    let mut changed = good_graph.clone();
    changed.pointers[0] = 0x1004;
    graph_mutations.push(changed);
    let mut changed = good_graph;
    changed.bundle[3] = 385;
    graph_mutations.push(changed);
    for changed in graph_mutations {
        assert!(fixed_auto_bundle_graph_contract(descriptor, &changed, pointers, shape).is_err());
    }
}

#[test]
fn fixed_force_bias_contract_requires_disallowed_launch_to_fail() {
    let spec = fixed_force_spec("f32_exact", (12, 0), InferenceTile::F32Sm120TmaFmaM128N64)
        .expect("SM120 exact TMA no-bias spec");

    assert_eq!(
        fixed_force_classify_first_launch(&spec, true, Err("bias rejected".into())).unwrap(),
        FixedForceFirstLaunch::BiasContractRejected {
            contract: "no_bias",
            reason: "bias rejected".into(),
        }
    );
    let accepted_wrong_bias = fixed_force_classify_first_launch(&spec, true, Ok(()))
        .expect_err("a successful wrong-bias launch must fail the inventory");
    assert!(accepted_wrong_bias.contains("accepted disallowed bias=true"));
    assert!(accepted_wrong_bias.contains("no_bias"));

    assert_eq!(
        fixed_force_classify_first_launch(&spec, false, Ok(())).unwrap(),
        FixedForceFirstLaunch::Runnable
    );
    assert_eq!(
        fixed_force_classify_first_launch(&spec, false, Err("holder absent".into())).unwrap(),
        FixedForceFirstLaunch::AvailabilityRejected("holder absent".into())
    );

    let with_bias = fixed_force_spec(
        "f32_exact",
        (12, 0),
        InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64,
    )
    .expect("SM120 exact TMA post-bias spec");
    assert_eq!(
        fixed_force_classify_first_launch(&with_bias, false, Err("bias required".into())).unwrap(),
        FixedForceFirstLaunch::BiasContractRejected {
            contract: "with_bias",
            reason: "bias required".into(),
        }
    );
    assert!(fixed_force_classify_first_launch(&with_bias, false, Ok(())).is_err());
}

#[test]
fn fixed_force_disallowed_candidate_is_invoked_exactly_once() {
    let spec = fixed_force_spec("f32_exact", (12, 0), InferenceTile::F32Sm120TmaFmaM128N64)
        .expect("SM120 exact TMA no-bias spec");
    let launches = std::cell::Cell::new(0);
    let disposition = fixed_force_run_first_launch(&spec, true, || {
        launches.set(launches.get() + 1);
        Err("bias rejected".into())
    })
    .unwrap();
    assert_eq!(launches.get(), 1);
    assert!(matches!(
        disposition,
        FixedForceFirstLaunch::BiasContractRejected { .. }
    ));
}

#[test]
fn fixed_force_specs_cover_every_inventory_entry_exactly_once() {
    for cc in [(8, 9), (12, 0)] {
        for row in [
            "bf16",
            "f16",
            "bf16_f32",
            "f16_f32",
            "tf32",
            "f32_exact",
            "f32_exact_fast",
        ] {
            let inventory = fixed_explicit_vendor_tiles(row, cc);
            let mut symbols = std::collections::BTreeSet::new();
            for tile in inventory.iter().copied() {
                let spec = fixed_force_spec(row, cc, tile)
                    .unwrap_or_else(|error| panic!("{row}/CC{cc:?}/{tile:?}: {error}"));
                assert_eq!(spec.tile, tile);
                assert_eq!(spec.row, row);
                assert_eq!(spec.compute_capability, cc);
                assert_eq!(
                    (spec.input_dtype, spec.output_dtype),
                    fixed_force_row_dtypes(row).unwrap()
                );
                assert!(
                    symbols.insert(spec.expected_symbol),
                    "duplicate force symbol in {row}/CC{cc:?}: {}",
                    spec.expected_symbol
                );
            }
            assert_eq!(symbols.len(), inventory.len());
        }
    }
}

#[test]
fn fixed_force_enum_registry_is_reverse_complete_for_every_row_and_architecture() {
    let universe = fixed_force_tile_universe();
    let mut unique = Vec::new();
    for tile in universe.iter().copied() {
        assert!(
            !unique.contains(&tile),
            "duplicate InferenceTile registry entry {tile:?}"
        );
        unique.push(tile);
    }
    let registered_halves = universe
        .iter()
        .filter_map(|tile| match tile {
            InferenceTile::Sm120Half(half) => Some(*half),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        registered_halves.len(),
        InferenceSm120HalfTile::ALL.len(),
        "InferenceSm120HalfTile::ALL must exactly match the exhaustive force registry"
    );
    for half in InferenceSm120HalfTile::ALL {
        assert!(
            registered_halves.contains(&half),
            "nested SM120 half tile {half:?} escaped the force registry"
        );
    }

    for tile in universe {
        for cc in [(8, 9), (12, 0)] {
            for row in [
                "bf16",
                "f16",
                "bf16_f32",
                "f16_f32",
                "tf32",
                "f32_exact",
                "f32_exact_fast",
            ] {
                let eligible = fixed_force_spec(row, cc, tile).is_ok();
                let occurrences = fixed_explicit_vendor_tiles(row, cc)
                    .into_iter()
                    .filter(|candidate| *candidate == tile)
                    .count();
                assert_eq!(
                    occurrences,
                    usize::from(eligible),
                    "reverse force inventory mismatch for {tile:?}/{row}/CC{cc:?}"
                );
            }
        }
    }
}

#[test]
fn fixed_force_specs_reject_foreign_arch_and_dtype_pairs() {
    assert!(fixed_force_spec("bf16", (12, 0), InferenceTile::Tc128Sm89Pipeline).is_err());
    assert!(
        fixed_force_spec(
            "bf16_f32",
            (8, 9),
            InferenceTile::Sm120Half(InferenceSm120HalfTile::M64N64Bk64S2)
        )
        .is_err()
    );
    assert!(fixed_force_spec("tf32", (12, 0), InferenceTile::Tf32M128N128S3).is_err());
    assert!(fixed_force_spec("bf16", (8, 9), InferenceTile::TcW64).is_ok());
    assert!(fixed_force_spec("bf16_f32", (8, 9), InferenceTile::TcW64).is_err());
}

#[test]
fn fixed_explicit_vendor_tile_filter_is_strict_and_row_scoped() {
    let inventory = [InferenceTile::Tc128, InferenceTile::Tc128Sm89Pipeline];
    assert_eq!(
        fixed_explicit_vendor_filter_tiles(&inventory, None).unwrap(),
        [InferenceTile::Tc128, InferenceTile::Tc128Sm89Pipeline]
    );
    assert_eq!(
        fixed_explicit_vendor_filter_tiles(&inventory, Some("Tc128Sm89Pipeline")).unwrap(),
        [InferenceTile::Tc128Sm89Pipeline]
    );
    assert_eq!(
        fixed_explicit_vendor_filter_tiles(&inventory, Some("Tc128Sm89Pipeline,Tc128")).unwrap(),
        [InferenceTile::Tc128Sm89Pipeline, InferenceTile::Tc128]
    );
    for invalid in [
        "",
        "tc128",
        "Tc128Sm89Pipeline ",
        " Tc128Sm89Pipeline",
        "Tc128,Tc128",
        "Tc128,",
        ",Tc128",
        "Tc64",
        "Tf32M128S2",
        "Sm120Half(M128N128Bk32S2)",
    ] {
        assert!(
            fixed_explicit_vendor_filter_tiles(&inventory, Some(invalid)).is_err(),
            "filter accepted unavailable, duplicate, or malformed tile {invalid:?}"
        );
    }
    for (row, cc) in [
        ("bf16", (12, 0)),
        ("f16", (12, 0)),
        ("tf32", (8, 9)),
        ("f32_exact", (8, 9)),
    ] {
        assert!(
            fixed_explicit_vendor_filter_tiles(
                &fixed_explicit_vendor_tiles(row, cc),
                Some("Tc128Sm89Pipeline")
            )
            .is_err(),
            "pipeline filter escaped its Ada homogeneous-half inventory: {row}/{cc:?}"
        );
    }
}

#[test]
fn fixed_explicit_vendor_paths_default_to_full_eager_and_graph() {
    assert_eq!(
        fixed_explicit_vendor_paths(None).unwrap(),
        ["eager", "graph"]
    );
    assert_eq!(
        fixed_explicit_vendor_paths(Some("eager")).unwrap(),
        ["eager"]
    );
    assert_eq!(
        fixed_explicit_vendor_paths(Some("graph")).unwrap(),
        ["graph"]
    );
    assert_eq!(
        fixed_explicit_vendor_paths(Some("eager,graph")).unwrap(),
        ["eager", "graph"]
    );
    assert_eq!(
        fixed_explicit_vendor_paths(Some("graph,eager")).unwrap(),
        ["graph", "eager"]
    );
    for invalid in [
        "",
        "Graph",
        "graph ",
        " eager",
        "eager,eager",
        "graph,",
        ",graph",
        "both",
    ] {
        assert!(
            fixed_explicit_vendor_paths(Some(invalid)).is_err(),
            "path filter accepted {invalid:?}"
        );
    }
}

#[test]
fn fixed_explicit_vendor_pipeline_graph_contract_rejects_wrong_physical_launch() {
    let shape = InferenceShape {
        m: 4621,
        k: 384,
        n: 1928,
    };
    let grid = (592, 1, 1);
    let abi = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
    let pointers = [0x1000, 0x2000, 0x3000, 0x4000];
    let bundle = [1.0f32.to_bits(), 0, 4621, 1928, 384, 384, 1928, 1928];
    let valid = |tile, dtype, nodes, symbol, grid, block, shared, abi, sixth, actual, params| {
        fixed_explicit_vendor_pipeline_graph_contract(
            tile,
            dtype,
            nodes,
            ObservedGemmNode {
                symbol,
                grid,
                block,
                shared_bytes: shared,
                driver_abi: abi,
                terminal_sixth_rejected: sixth,
                pointers: actual,
                bundle: params,
            },
            pointers,
            shape,
        )
    };
    for (tile, dtype, symbol, required_shared) in [
        (
            InferenceTile::Tc128Sm89Pipeline,
            WeightDtype::Bf16,
            "nn_sm89_tc128_pipeline_bf16",
            71_680,
        ),
        (
            InferenceTile::Tc128Sm89Pipeline,
            WeightDtype::F16,
            "nn_sm89_tc128_pipeline_f16",
            71_680,
        ),
        (
            InferenceTile::Tc128Sm89Swizzle,
            WeightDtype::Bf16,
            "nn_sm89_tc128_swizzle_bf16",
            69_632,
        ),
        (
            InferenceTile::Tc128Sm89Swizzle,
            WeightDtype::F16,
            "nn_sm89_tc128_swizzle_f16",
            69_632,
        ),
        (
            InferenceTile::Tc128Sm89S3,
            WeightDtype::Bf16,
            "nn_sm89_tc128_s3_bf16",
            98_304,
        ),
        (
            InferenceTile::Tc128Sm89S3,
            WeightDtype::F16,
            "nn_sm89_tc128_s3_f16",
            98_304,
        ),
    ] {
        valid(
            tile,
            dtype,
            1,
            symbol,
            grid,
            (256, 1, 1),
            required_shared,
            abi.to_vec(),
            true,
            pointers,
            bundle,
        )
        .expect("one actual dtype-specific pipeline launch");
        for (count, block, shared) in [
            (0, (256, 1, 1), required_shared),
            (2, (256, 1, 1), required_shared),
            (1, (128, 1, 1), required_shared),
            (1, (256, 2, 1), required_shared),
            (1, (256, 1, 2), required_shared),
            (1, (256, 1, 1), 0),
            (1, (256, 1, 1), required_shared + 1),
        ] {
            assert!(
                valid(
                    tile,
                    dtype,
                    count,
                    symbol,
                    grid,
                    block,
                    shared,
                    abi.to_vec(),
                    true,
                    pointers,
                    bundle,
                )
                .is_err(),
                "accepted pipeline launch count={count} block={block:?} shared={shared}"
            );
        }
        for wrong_symbol in ["nn_tc128_bf16", "", "nn_sm89_tc128_pipeline_decoy_bf16"] {
            assert!(
                valid(
                    tile,
                    dtype,
                    1,
                    wrong_symbol,
                    grid,
                    (256, 1, 1),
                    71_680,
                    abi.to_vec(),
                    true,
                    pointers,
                    bundle,
                )
                .is_err(),
                "accepted wrong pipeline symbol {wrong_symbol:?}"
            );
        }
        let other_dtype = if dtype == WeightDtype::Bf16 {
            WeightDtype::F16
        } else {
            WeightDtype::Bf16
        };
        assert!(
            valid(
                tile,
                other_dtype,
                1,
                symbol,
                grid,
                (256, 1, 1),
                71_680,
                abi.to_vec(),
                true,
                pointers,
                bundle,
            )
            .is_err(),
            "accepted the other homogeneous-half symbol"
        );
        assert!(
            valid(
                tile,
                WeightDtype::F32,
                1,
                symbol,
                grid,
                (256, 1, 1),
                71_680,
                abi.to_vec(),
                true,
                pointers,
                bundle,
            )
            .is_err(),
            "accepted F32 for a homogeneous-half pipeline"
        );
        for (bad_grid, bad_abi, sixth, bad_pointers, bad_bundle) in [
            ((591, 1, 1), abi.to_vec(), true, pointers, bundle),
            ((592, 2, 1), abi.to_vec(), true, pointers, bundle),
            (
                grid,
                vec![(0, 8), (8, 8), (16, 8), (24, 8)],
                true,
                pointers,
                bundle,
            ),
            (
                grid,
                vec![(0, 8), (8, 8), (16, 4), (24, 8), (32, 32)],
                true,
                pointers,
                bundle,
            ),
            (grid, abi.to_vec(), false, pointers, bundle),
            (
                grid,
                abi.to_vec(),
                true,
                [0x1001, 0x2000, 0x3000, 0x4000],
                bundle,
            ),
            (
                grid,
                abi.to_vec(),
                true,
                pointers,
                [1.0f32.to_bits(), 0, 4621, 384, 1928, 384, 1928, 1928],
            ),
        ] {
            assert!(
                valid(
                    tile,
                    dtype,
                    1,
                    symbol,
                    bad_grid,
                    (256, 1, 1),
                    required_shared,
                    bad_abi,
                    sixth,
                    bad_pointers,
                    bad_bundle,
                )
                .is_err(),
                "accepted mutated Ada half grid/ABI/pointer/bundle"
            );
        }
    }
}

#[test]
#[ignore = "requires MAMBA_FIXED_ADA_VENDOR=1 and an explicitly admitted quiet CC8.9/12.0 GPU (default 8.9); emits paired production AUTO evidence"]
fn fixed_ada_production_auto_paired_precision_cublas() {
    use cudarc::cublas::sys::cublasComputeType_t;

    assert_eq!(
        std::env::var("MAMBA_FIXED_ADA_VENDOR").as_deref(),
        Ok("1"),
        "set MAMBA_FIXED_ADA_VENDOR=1 to run the explicit vendor comparator"
    );
    assert_ne!(
        std::env::var("NVIDIA_TF32_OVERRIDE").as_deref(),
        Ok("0"),
        "NVIDIA_TF32_OVERRIDE=0 disables the explicit FAST_TF32 denominator"
    );
    if cfg!(debug_assertions) {
        panic!("explicit vendor comparator requires --release");
    }
    let rows = fixed_explicit_vendor_row_specs();
    let selected_rows = fixed_ada_filter("MAMBA_FIXED_ADA_ROWS", &rows.map(|row| row.name));
    let labels = FIXED_AUTO_VENDOR_EXACT_CELLS
        .iter()
        .map(|cell| cell.label)
        .collect::<Vec<_>>();
    let selected_cells = fixed_ada_filter("MAMBA_FIXED_ADA_CELLS", &labels);
    let biases = fixed_ada_filter("MAMBA_FIXED_ADA_BIAS", &["0", "1"]);
    let requested_paths = match std::env::var("MAMBA_FIXED_VENDOR_PATHS") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("read MAMBA_FIXED_VENDOR_PATHS: {error}"),
    };
    let paths = fixed_explicit_vendor_paths(requested_paths.as_deref())
        .expect("explicit vendor path filter");
    let requested_tiles = match std::env::var("MAMBA_FIXED_VENDOR_TILES") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("read MAMBA_FIXED_VENDOR_TILES: {error}"),
    };
    fixed_ada_direct_pair_reject_vendor_tiles(requested_tiles.as_deref())
        .expect("AUTO-only comparator forbids forced-tile filters");
    let windows = match std::env::var("MAMBA_FIXED_ADA_WINDOWS") {
        Ok(value) => value
            .parse::<usize>()
            .expect("MAMBA_FIXED_ADA_WINDOWS must be an integer"),
        Err(std::env::VarError::NotPresent) => FIXED_PRODUCTION_AUTO_DEFAULT_WINDOWS,
        Err(error) => panic!("read MAMBA_FIXED_ADA_WINDOWS: {error}"),
    };
    assert!(
        (1..=10_001).contains(&windows),
        "explicit vendor windows must be in 1..=10001"
    );
    let state_capacity = state_capacity_from_env().expect("parse production AUTO state capacity");
    let quiet_gpu = QuietGpu::for_cuda_ordinal(0).expect("resolve CUDA device 0 UUID");
    let _pre_context = quiet_gpu
        .require_pre_context("inference-production-auto/pre-context")
        .expect("exclusive CUDA device 0 before context creation");
    fixed_sm120_tf32_bd_environment_preflight("explicit vendor AUTO/vendor")
        .expect("explicit vendor AUTO/vendor preflight");
    let requested_cc = match std::env::var("MAMBA_FIXED_VENDOR_EXACT_CC") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("read MAMBA_FIXED_VENDOR_EXACT_CC: {error}"),
    };
    let device = GpuDevice::new(0).expect("explicit vendor CUDA device");
    fixed_explicit_vendor_admit_cc(requested_cc.as_deref(), device.compute_capability)
        .expect("explicit vendor exact-CC admission");
    assert!(
        matches!(
            (device.compute_capability, device.multiprocessor_count()),
            ((8, 9), 142) | ((12, 0), 170)
        ),
        "explicit vendor final evidence supports only CC8.9/142SM or CC12.0/170SM"
    );
    let ctx =
        GpuCtx::new_with_state_cap(&device, state_capacity).expect("explicit vendor GPU context");
    assert_eq!(
        ctx.state_cap(),
        state_capacity,
        "production AUTO context ignored requested state capacity"
    );
    let cohort = render_cohort_fragment(
        ProductionAutoInventory::Release070,
        ctx.state_cap(),
        ctx._blas_workspace.len(),
    );
    let compiler = ctx.kernels.compiler_identity();
    assert_eq!(
        compiler.nvrtc_version,
        (13, 2),
        "final production AUTO evidence is CUDA 13.2 only"
    );
    assert!(
        compiler.nvrtc_library_known,
        "final production AUTO evidence requires a known NVRTC library"
    );
    let fixed_artifact = ctx.kernels.artifact_set_identity().fixed;
    let device_identity = device.identity();
    let git_sha = std::env::var("GEMM_BI_FINAL_AUTO_GIT_SHA")
        .expect("GEMM_BI_FINAL_AUTO_GIT_SHA must bind the final source");
    assert!(
        git_sha.len() == 40 && git_sha.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "GEMM_BI_FINAL_AUTO_GIT_SHA must be exactly 40 hexadecimal characters"
    );
    let gpu_uuid = fixed_production_auto_cuda_uuid(0)
        .expect("resolve final production AUTO CUDA ordinal 0 UUID");
    let cuda_home = std::env::var("CUDA_HOME").expect("CUDA_HOME must identify the final toolkit");
    let ld_library_path = std::env::var("LD_LIBRARY_PATH")
        .expect("LD_LIBRARY_PATH must identify loaded CUDA libraries");
    let device_metadata = format!(
        concat!(
            "\"git_sha\":\"{}\",\"gpu_uuid\":\"{}\",{}, {},",
            "\"cc\":\"{}.{}\",\"sm_count\":{},\"nvrtc\":[{},{}],",
            "\"compiler_target\":\"{:?}\",\"cuda_home\":\"{}\",",
            "\"ld_library_path\":\"{}\",\"fixed_source_digest\":\"{}\",",
            "\"fixed_invocation_digest\":\"{}\",\"fixed_artifact_digest\":\"{}\",",
            "\"header_manifest_digest\":\"{}\",\"nvrtc_library_domain\":\"{}\",",
            "\"nvrtc_library_known\":{},\"driver_api_version\":{},",
            "\"driver_build_sources\":{},\"driver_build_digest\":\"{}\""
        ),
        fixed_sm120_tf32_bd_json_escape(&git_sha),
        fixed_sm120_tf32_bd_json_escape(&gpu_uuid),
        fixed_production_auto_tuning_metadata(),
        cohort,
        device.compute_capability.0,
        device.compute_capability.1,
        device.multiprocessor_count(),
        compiler.nvrtc_version.0,
        compiler.nvrtc_version.1,
        compiler.target,
        fixed_sm120_tf32_bd_json_escape(&cuda_home),
        fixed_sm120_tf32_bd_json_escape(&ld_library_path),
        digest_hex(&compiler.source_digest),
        digest_hex(&compiler.invocation_digest),
        digest_hex(&fixed_artifact.artifact_digest),
        digest_hex(&compiler.header_manifest_digest),
        digest_hex(&compiler.nvrtc_library_domain),
        compiler.nvrtc_library_known,
        device_identity.driver.api_version,
        device_identity.driver.build_sources,
        digest_hex(&device_identity.driver.build_digest),
    );
    let full_inventory = selected_rows.len() == rows.len()
        && selected_cells.len() == FIXED_AUTO_VENDOR_EXACT_CELLS.len()
        && biases.len() == 2
        && paths.len() == 2;
    let selected_row_count = selected_rows.len();
    let selected_cell_count = selected_cells.len();
    let selected_bias_count = biases.len();
    let selected_path_count = paths.len();
    let expected_records = fixed_production_auto_expected_records(
        selected_row_count,
        selected_cell_count,
        selected_bias_count,
        selected_path_count,
    );
    let mut records = 0usize;
    for row_index in selected_rows {
        let row_spec = rows[row_index];
        configure_fixed_auto_vendor_custom(&ctx, row_spec.policy);
        for &cell_index in &selected_cells {
            let cell = FIXED_AUTO_VENDOR_EXACT_CELLS[cell_index];
            let shape = cell.shape;
            let elements = shape.m * shape.n;
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, row_spec.input_dtype)
                .expect("explicit vendor A");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, row_spec.input_dtype)
                .expect("explicit vendor B");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x0ada_a001))
                .expect("explicit vendor A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x0ada_b001))
                .expect("explicit vendor B upload");
            let auto = DtypedBuf::zeros(&ctx.stream, elements, row_spec.output_dtype)
                .expect("explicit vendor AUTO C");
            let vendor = DtypedBuf::zeros(&ctx.stream, elements, row_spec.output_dtype)
                .expect("explicit vendor vendor C");
            let reference = DtypedBuf::zeros(&ctx.stream, elements, WeightDtype::F32)
                .expect("explicit vendor reference C");
            let bias = DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32)
                .expect("explicit vendor bias");
            bias.upload_f32(&ctx.stream, &synth(shape.n, 0x0ada_b1a5))
                .expect("explicit vendor bias upload");
            for &bias_index in &biases {
                let has_bias = bias_index == 1;
                let auto_ops = InferenceFwdOperands {
                    c: typed(&auto, row_spec.output_dtype),
                    x: typed(&a, row_spec.input_dtype),
                    w: typed(&b, row_spec.input_dtype),
                    bias_ptr: has_bias.then(|| bias.cached_ptr()),
                };
                let vendor_ops = InferenceFwdOperands {
                    c: typed(&vendor, row_spec.output_dtype),
                    ..auto_ops
                };
                let reference_ops = InferenceFwdOperands {
                    c: typed(&reference, WeightDtype::F32),
                    ..auto_ops
                };
                let mut recorded_selected = None;
                let auto_trace = ctx
                    .record_eager_gemm_trace(|| {
                        recorded_selected =
                            Some(launch_fixed_auto_vendor_custom(&ctx, auto_ops, shape));
                        Ok(())
                    })
                    .unwrap_or_else(|error| {
                        panic!(
                            "record production AUTO physical route {}/{} bias={has_bias}: {error}",
                            row_spec.name, cell.label
                        )
                    });
                let selected = recorded_selected.unwrap_or_else(|| {
                    panic!(
                        "production AUTO physical route did not return a family for {}/{} bias={has_bias}",
                        row_spec.name, cell.label
                    )
                });
                let recorded_routes = auto_trace
                    .routes()
                    .iter()
                    .map(FixedAutoRecordedRoute::from)
                    .collect::<Vec<_>>();
                let auto_descriptor = fixed_auto_bundle_physical_descriptor(
                    FixedAutoPhysicalRequest {
                        row: row_spec.name,
                        operands: auto_ops,
                        shape,
                        selected,
                        compute_capability: device.compute_capability,
                        multiprocessors: device.multiprocessor_count(),
                        compiler_target: compiler.target.as_str(),
                        state_capacity: ctx.state_cap(),
                        nvrtc: compiler.nvrtc_version,
                        nvrtc_library_known: compiler.nvrtc_library_known,
                        policy: row_spec.policy,
                    },
                    &recorded_routes,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "resolve production AUTO physical route {}/{} bias={has_bias}: {error}",
                        row_spec.name, cell.label
                    )
                });
                let auto_bits = f32_bits(&ctx, &auto, elements);
                let auto_raw = fixed_explicit_vendor_raw_bytes(&ctx, &auto);
                assert_eq!(
                    launch_fixed_auto_vendor_custom(&ctx, auto_ops, shape),
                    selected,
                    "production AUTO eager tile changed"
                );
                assert_eq!(
                    fixed_explicit_vendor_raw_bytes(&ctx, &auto),
                    auto_raw,
                    "production AUTO eager repeat bits changed"
                );
                fixed_ada_vendor_launch(
                    &ctx,
                    reference_ops,
                    shape,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
                );
                let reference_bits = f32_bits(&ctx, &reference, elements);
                let auto_error = fixed_ada_normalized_error(
                    &auto_bits,
                    &reference_bits,
                    row_spec.custom_tolerance,
                    &format!(
                        "production AUTO {}/{} bias={has_bias}",
                        row_spec.name, cell.label
                    ),
                );
                fixed_ada_vendor_launch(&ctx, vendor_ops, shape, row_spec.vendor_compute);
                let vendor_bits = f32_bits(&ctx, &vendor, elements);
                let vendor_raw = fixed_explicit_vendor_raw_bytes(&ctx, &vendor);
                let vendor_error = fixed_ada_normalized_error(
                    &vendor_bits,
                    &reference_bits,
                    row_spec.vendor_tolerance,
                    &format!(
                        "vendor {} {}/{} bias={has_bias}",
                        row_spec.vendor_comparator, row_spec.name, cell.label
                    ),
                );
                fixed_ada_vendor_launch(&ctx, vendor_ops, shape, row_spec.vendor_compute);
                assert_eq!(
                    fixed_explicit_vendor_raw_bytes(&ctx, &vendor),
                    vendor_raw,
                    "explicit vendor eager repeat bits changed"
                );
                let auto_launch = || {
                    assert_eq!(
                        launch_fixed_auto_vendor_custom(&ctx, auto_ops, shape),
                        selected,
                        "production AUTO tile changed during final measurement"
                    );
                };
                let vendor_launch =
                    || fixed_ada_vendor_launch(&ctx, vendor_ops, shape, row_spec.vendor_compute);
                for _ in 0..128 {
                    auto_launch();
                    vendor_launch();
                }
                ctx.stream
                    .synchronize()
                    .expect("production AUTO/vendor eager warmup");
                let auto_graph = unsafe {
                    capture_into_graph(&ctx.stream, || {
                        auto_launch();
                        Ok(())
                    })
                }
                .expect("capture complete production AUTO graph");
                let vendor_graph = unsafe {
                    capture_into_graph(&ctx.stream, || {
                        vendor_launch();
                        Ok(())
                    })
                }
                .expect("capture complete explicit vendor graph");
                let auto_inventory = fixed_explicit_vendor_graph_inventory(
                    &auto_graph,
                    "AUTO",
                    (auto_descriptor.is_none() && fixed_explicit_vendor_ada_descriptor(selected))
                        .then_some((selected, row_spec.input_dtype, auto_ops, shape)),
                    auto_descriptor.map(|descriptor| (descriptor, auto_ops, shape)),
                    None,
                );
                let expected_auto_symbol = match auto_descriptor {
                    Some(descriptor) => descriptor.symbol,
                    None => fixed_force_spec(
                        row_spec.name,
                        device.compute_capability,
                        selected,
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "resolve legacy production AUTO physical route {}/{} bias={has_bias}: {error}",
                            row_spec.name, cell.label
                        )
                    })
                    .expected_symbol,
                };
                assert!(
                    fixed_production_auto_inventory_has_exact_symbol(
                        &auto_inventory,
                        expected_auto_symbol,
                    ),
                    "captured AUTO graph omitted selected symbol {expected_auto_symbol}"
                );
                let bias_symbol = has_bias.then_some(match row_spec.output_dtype {
                    WeightDtype::F32 => "bias_broadcast",
                    WeightDtype::Bf16 => "bias_broadcast_bf16",
                    WeightDtype::F16 => "bias_broadcast_f16",
                });
                let vendor_inventory = fixed_explicit_vendor_graph_inventory(
                    &vendor_graph,
                    "vendor",
                    None,
                    None,
                    bias_symbol,
                );
                let graph_inventory =
                    format!("{{\"auto\":{auto_inventory},\"vendor\":{vendor_inventory}}}");
                for &path in &paths {
                    let auto_run = || {
                        if path == "graph" {
                            auto_graph.launch().expect("production AUTO graph replay");
                        } else {
                            auto_launch();
                        }
                    };
                    let vendor_run = || {
                        if path == "graph" {
                            vendor_graph.launch().expect("explicit vendor graph replay");
                        } else {
                            vendor_launch();
                        }
                    };
                    if path == "graph" {
                        for replay in 0..2 {
                            fixed_explicit_vendor_poison_output(&ctx, &auto);
                            auto_run();
                            assert_eq!(
                                fixed_explicit_vendor_raw_bytes(&ctx, &auto),
                                auto_raw,
                                "production AUTO graph replay {replay} changed storage bits"
                            );
                            fixed_explicit_vendor_poison_output(&ctx, &vendor);
                            vendor_run();
                            assert_eq!(
                                fixed_explicit_vendor_raw_bytes(&ctx, &vendor),
                                vendor_raw,
                                "explicit vendor graph replay {replay} changed storage bits"
                            );
                        }
                    }
                    for _ in 0..128 {
                        auto_run();
                        vendor_run();
                    }
                    ctx.stream
                        .synchronize()
                        .expect("production AUTO/vendor path warmup");
                    let mut iterations = None;
                    for auto_first in [true, false] {
                        let order = if auto_first {
                            "auto_then_vendor"
                        } else {
                            "vendor_then_auto"
                        };
                        let cohort_label = format!(
                            "inference-production-auto/{}/{}/bias={}/{path}/{order}",
                            row_spec.name, cell.label, has_bias
                        );
                        let quiet_preflight = quiet_gpu
                            .require_cohort(&cohort_label)
                            .expect("quiet GPU before production AUTO cohort");
                        let (auto_iterations, vendor_iterations) = match iterations {
                            Some(iterations) => iterations,
                            None => {
                                let calibrated = (
                                    fixed_auto_vendor_iterations(fixed_ada_event_window_us(
                                        &ctx, 16, auto_run,
                                    )),
                                    fixed_auto_vendor_iterations(fixed_ada_event_window_us(
                                        &ctx, 16, vendor_run,
                                    )),
                                );
                                iterations = Some(calibrated);
                                calibrated
                            }
                        };
                        let mut auto_samples = Vec::with_capacity(windows);
                        let mut vendor_samples = Vec::with_capacity(windows);
                        for _ in 0..windows {
                            let (auto_us, vendor_us) = if auto_first {
                                (
                                    fixed_ada_event_window_us(&ctx, auto_iterations, auto_run),
                                    fixed_ada_event_window_us(&ctx, vendor_iterations, vendor_run),
                                )
                            } else {
                                let vendor_us =
                                    fixed_ada_event_window_us(&ctx, vendor_iterations, vendor_run);
                                (
                                    fixed_ada_event_window_us(&ctx, auto_iterations, auto_run),
                                    vendor_us,
                                )
                            };
                            auto_samples.push(auto_us);
                            vendor_samples.push(vendor_us);
                        }
                        let ratios =
                            fixed_production_auto_ratio_samples(&auto_samples, &vendor_samples)
                                .expect("valid paired AUTO/vendor samples");
                        assert_eq!(
                            fixed_explicit_vendor_raw_bytes(&ctx, &auto),
                            auto_raw,
                            "production AUTO post-timing bits changed"
                        );
                        assert_eq!(
                            fixed_explicit_vendor_raw_bytes(&ctx, &vendor),
                            vendor_raw,
                            "explicit vendor post-timing bits changed"
                        );
                        ctx.stream
                            .synchronize()
                            .expect("complete production AUTO paired timing");
                        let quiet_postflight = quiet_gpu
                            .verify_post_cohort(&cohort_label)
                            .expect("quiet GPU after production AUTO cohort");
                        let mut auto_sorted = auto_samples.clone();
                        let mut vendor_sorted = vendor_samples.clone();
                        let mut ratio_sorted = ratios.clone();
                        auto_sorted.sort_by(f64::total_cmp);
                        vendor_sorted.sort_by(f64::total_cmp);
                        ratio_sorted.sort_by(f64::total_cmp);
                        println!(
                            concat!(
                                "{{\"schema\":\"MambaBiFixedFinalProductionAutoVendorV1\",{},",
                                "\"scope\":\"performance_only\",\"call_scope\":\"production_auto\",",
                                "\"row\":\"{}\",\"cell\":\"{}\",\"m\":{},\"k\":{},\"n\":{},",
                                "\"input_dtype\":\"{}\",\"output_dtype\":\"{}\",\"auto_tile\":\"{:?}\",",
                                "\"op\":\"nn\",\"path\":\"{}\",\"graphs\":{},",
                                "\"graph_replay_bits_equal\":{},\"raw_storage_bits_equal\":true,",
                                "\"eager_repeat_bits_equal\":true,\"vendor_repeat_bits_equal\":true,",
                                "\"timing\":\"cuda_events\",\"alpha\":1,\"beta\":0,\"bias\":{},",
                                "\"vendor_gemm_beta\":{},\"vendor_bias_broadcast_timed\":{},",
                                "{},\"vendor_compute\":\"{:?}\",",
                                "\"reference_compute\":\"CUBLAS_COMPUTE_32F_PEDANTIC\",",
                                "\"reference_output_dtype\":\"f32\",\"auto_normalized_error\":{},",
                                "\"vendor_normalized_error\":{},\"order\":\"{}\",\"windows\":{},",
                                "\"auto_iterations\":{},\"vendor_iterations\":{},",
                                "\"auto_p50_us\":{},\"auto_p95_us\":{},",
                                "\"vendor_p50_us\":{},\"vendor_p95_us\":{},",
                                "\"auto_over_vendor_p50\":{},\"auto_over_vendor_p95\":{},",
                                "\"auto_samples_us\":{:?},\"vendor_samples_us\":{:?},",
                                "\"auto_over_vendor_samples\":{:?},",
                                "\"quiet_preflight\":\"{}\",\"quiet_postflight\":\"{}\"}}"
                            ),
                            device_metadata,
                            row_spec.name,
                            cell.label,
                            shape.m,
                            shape.k,
                            shape.n,
                            row_spec.input_dtype.as_str(),
                            row_spec.output_dtype.as_str(),
                            selected,
                            path,
                            graph_inventory,
                            path == "graph",
                            has_bias,
                            u8::from(has_bias),
                            has_bias,
                            fixed_explicit_vendor_tolerance_metadata(&row_spec),
                            row_spec.vendor_compute,
                            auto_error,
                            vendor_error,
                            order,
                            windows,
                            auto_iterations,
                            vendor_iterations,
                            percentile(&auto_sorted, 0.50),
                            percentile(&auto_sorted, 0.95),
                            percentile(&vendor_sorted, 0.50),
                            percentile(&vendor_sorted, 0.95),
                            percentile(&ratio_sorted, 0.50),
                            percentile(&ratio_sorted, 0.95),
                            auto_samples,
                            vendor_samples,
                            ratios,
                            fixed_sm120_tf32_bd_json_escape(&quiet_preflight),
                            fixed_sm120_tf32_bd_json_escape(&quiet_postflight),
                        );
                        records += 1;
                    }
                }
            }
        }
    }
    assert_eq!(
        records, expected_records,
        "production AUTO/vendor record count"
    );
    if full_inventory {
        assert_eq!(records, 280, "full production AUTO/vendor record count");
    }
    println!(
        "{{\"schema\":\"MambaBiFixedFinalProductionAutoVendorCompletionV1\",{}, {},\"scope\":\"performance_only\",\"full_inventory\":{},\"rows\":{},\"cells\":{},\"biases\":{},\"paths\":{},\"orders\":2,\"windows_per_order\":{},\"records\":{},\"passed\":true}}",
        device_metadata,
        fixed_exact_comparator_completion_metadata(),
        full_inventory,
        selected_row_count,
        selected_cell_count,
        selected_bias_count,
        selected_path_count,
        windows,
        records,
    );
}
#[test]
#[ignore = "requires MAMBA_FIXED_ADA_VENDOR=1 and an explicitly admitted quiet CC8.9/12.0 GPU (default 8.9); emits forced-rung/AUTO/vendor evidence"]
fn fixed_ada_forced_rungs_paired_precision_cublas() {
    use cudarc::cublas::sys::cublasComputeType_t;

    match std::env::var("MAMBA_FIXED_ADA_TOOLKIT_ADMISSION") {
        Ok(value) => {
            assert_eq!(
                value, "1",
                "MAMBA_FIXED_ADA_TOOLKIT_ADMISSION must be exactly 1 when present"
            );
            toolkit_admission::run();
            return;
        }
        Err(std::env::VarError::NotPresent) => {}
        Err(error) => panic!("read MAMBA_FIXED_ADA_TOOLKIT_ADMISSION: {error}"),
    }

    assert_eq!(
        std::env::var("MAMBA_FIXED_ADA_VENDOR").as_deref(),
        Ok("1"),
        "set MAMBA_FIXED_ADA_VENDOR=1 to run the explicit vendor rung survey"
    );
    assert_ne!(
        std::env::var("NVIDIA_TF32_OVERRIDE").as_deref(),
        Ok("0"),
        "NVIDIA_TF32_OVERRIDE=0 disables the explicit FAST_TF32 denominator"
    );
    if cfg!(debug_assertions) {
        panic!("the explicit vendor rung survey requires --release");
    }
    let rows = fixed_explicit_vendor_row_specs();
    let selected_rows = fixed_ada_filter("MAMBA_FIXED_ADA_ROWS", &rows.map(|row| row.name));
    let labels = FIXED_AUTO_VENDOR_EXACT_CELLS
        .iter()
        .map(|cell| cell.label)
        .collect::<Vec<_>>();
    let selected_cells = fixed_ada_filter("MAMBA_FIXED_ADA_CELLS", &labels);
    let biases = fixed_ada_filter("MAMBA_FIXED_ADA_BIAS", &["0", "1"]);
    let windows = match std::env::var("MAMBA_FIXED_ADA_WINDOWS") {
        Ok(value) => value
            .parse::<usize>()
            .expect("MAMBA_FIXED_ADA_WINDOWS must be an integer"),
        Err(std::env::VarError::NotPresent) => 101,
        Err(error) => panic!("read MAMBA_FIXED_ADA_WINDOWS: {error}"),
    };
    assert!(
        (1..=10_001).contains(&windows),
        "explicit vendor windows must be in 1..=10001"
    );
    // Strict optional filters: exact Debug tile names, and eager/graph paths.
    // Absent filters preserve the full target-valid inventory and both paths.
    let requested_tiles = match std::env::var("MAMBA_FIXED_VENDOR_TILES") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("read MAMBA_FIXED_VENDOR_TILES: {error}"),
    };
    let requested_paths = match std::env::var("MAMBA_FIXED_VENDOR_PATHS") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("read MAMBA_FIXED_VENDOR_PATHS: {error}"),
    };
    let paths = fixed_explicit_vendor_paths(requested_paths.as_deref())
        .expect("explicit vendor path filter");
    fixed_sm120_tf32_bd_environment_preflight("explicit vendor forced-rung/AUTO/vendor")
        .expect("explicit vendor forced-rung preflight");
    let requested_cc = match std::env::var("MAMBA_FIXED_VENDOR_EXACT_CC") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("read MAMBA_FIXED_VENDOR_EXACT_CC: {error}"),
    };
    let device = GpuDevice::new(0).expect("explicit vendor CUDA device");
    fixed_explicit_vendor_admit_cc(requested_cc.as_deref(), device.compute_capability)
        .expect("explicit vendor exact-CC admission");
    let ctx = GpuCtx::new(&device).expect("explicit vendor GPU context");
    let compiler = ctx.kernels.compiler_identity();
    let device_metadata = format!(
        "\"cc\":\"{}.{}\",\"sm_count\":{},\"nvrtc\":[{},{}],\"compiler_target\":\"{:?}\"",
        device.compute_capability.0,
        device.compute_capability.1,
        device.multiprocessor_count(),
        compiler.nvrtc_version.0,
        compiler.nvrtc_version.1,
        compiler.target,
    );
    let fixed_artifact = ctx.kernels.artifact_set_identity().fixed;
    let device_metadata = format!(
        "{device_metadata},\"fixed_source_digest\":\"{}\",\"fixed_invocation_digest\":\"{}\",\"fixed_artifact_digest\":\"{}\",\"header_manifest_digest\":\"{}\",\"nvrtc_library_domain\":\"{}\",\"nvrtc_library_known\":{}",
        digest_hex(&compiler.source_digest),
        digest_hex(&compiler.invocation_digest),
        digest_hex(&fixed_artifact.artifact_digest),
        digest_hex(&compiler.header_manifest_digest),
        digest_hex(&compiler.nvrtc_library_domain),
        compiler.nvrtc_library_known,
    );
    let mut records = 0usize;
    let mut rejected = 0usize;
    for row_index in selected_rows {
        let row_spec = rows[row_index];
        let row = row_spec.name;
        let input_dtype = row_spec.input_dtype;
        let output_dtype = row_spec.output_dtype;
        let policy = row_spec.policy;
        let compute = row_spec.vendor_compute;
        let custom_tolerance = row_spec.custom_tolerance;
        let vendor_tolerance = row_spec.vendor_tolerance;
        let vendor_comparator = row_spec.vendor_comparator;
        let tiles = fixed_explicit_vendor_filter_tiles(
            &fixed_explicit_vendor_tiles(row, device.compute_capability),
            requested_tiles.as_deref(),
        )
        .unwrap_or_else(|error| panic!("{row}: {error}"));
        configure_fixed_auto_vendor_custom(&ctx, policy);
        for &cell_index in &selected_cells {
            let cell = FIXED_AUTO_VENDOR_EXACT_CELLS[cell_index];
            let shape = cell.shape;
            let elements = shape.m * shape.n;
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, input_dtype).expect("rung A");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, input_dtype).expect("rung B");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x0ada_a001))
                .expect("rung A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x0ada_b001))
                .expect("rung B upload");
            let auto = DtypedBuf::zeros(&ctx.stream, elements, output_dtype).expect("rung AUTO C");
            let forced =
                DtypedBuf::zeros(&ctx.stream, elements, output_dtype).expect("rung forced C");
            let vendor =
                DtypedBuf::zeros(&ctx.stream, elements, output_dtype).expect("rung vendor C");
            let reference = DtypedBuf::zeros(&ctx.stream, elements, WeightDtype::F32)
                .expect("rung reference C");
            let bias = DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32).expect("rung bias");
            bias.upload_f32(&ctx.stream, &synth(shape.n, 0x0ada_b1a5))
                .expect("rung bias upload");
            for &bias_index in &biases {
                let has_bias = bias_index == 1;
                let auto_ops = InferenceFwdOperands {
                    c: typed(&auto, output_dtype),
                    x: typed(&a, input_dtype),
                    w: typed(&b, input_dtype),
                    bias_ptr: has_bias.then(|| bias.cached_ptr()),
                };
                let forced_ops = InferenceFwdOperands {
                    c: typed(&forced, output_dtype),
                    ..auto_ops
                };
                let vendor_ops = InferenceFwdOperands {
                    c: typed(&vendor, output_dtype),
                    ..auto_ops
                };
                let reference_ops = InferenceFwdOperands {
                    c: typed(&reference, WeightDtype::F32),
                    ..auto_ops
                };
                let mut recorded_selected = None;
                let auto_trace = ctx
                    .record_eager_gemm_trace(|| {
                        recorded_selected =
                            Some(launch_fixed_auto_vendor_custom(&ctx, auto_ops, shape));
                        Ok(())
                    })
                    .unwrap_or_else(|error| {
                        panic!(
                            "record rung AUTO physical route {row}/{} bias={has_bias}: {error}",
                            cell.label
                        )
                    });
                let selected = recorded_selected.unwrap_or_else(|| {
                    panic!(
                        "rung AUTO physical route did not return a family for {row}/{} bias={has_bias}",
                        cell.label
                    )
                });
                let recorded_routes = auto_trace
                    .routes()
                    .iter()
                    .map(FixedAutoRecordedRoute::from)
                    .collect::<Vec<_>>();
                let auto_descriptor = fixed_auto_bundle_physical_descriptor(
                    FixedAutoPhysicalRequest {
                        row,
                        operands: auto_ops,
                        shape,
                        selected,
                        compute_capability: device.compute_capability,
                        multiprocessors: device.multiprocessor_count(),
                        compiler_target: compiler.target.as_str(),
                        state_capacity: ctx.state_cap(),
                        nvrtc: compiler.nvrtc_version,
                        nvrtc_library_known: compiler.nvrtc_library_known,
                        policy,
                    },
                    &recorded_routes,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "resolve rung AUTO physical route {row}/{} bias={has_bias}: {error}",
                        cell.label
                    )
                });
                if device.compute_capability == (8, 9)
                    && device.multiprocessor_count() == 142
                    && compiler.nvrtc_library_known
                {
                    let expected =
                        expected_ada_finalist_auto(compiler.nvrtc_version, row, shape, has_bias)
                            .or_else(|| {
                                (input_dtype == output_dtype && input_dtype.is_half())
                                    .then(|| {
                                        expected_ada_half_auto(
                                            compiler.nvrtc_version,
                                            input_dtype,
                                            shape,
                                            has_bias,
                                        )
                                    })
                                    .flatten()
                            });
                    if let Some(expected) = expected {
                        assert_eq!(
                            selected, expected,
                            "literal AUTO45 {row}/{} bias={has_bias}",
                            cell.label
                        );
                    }
                }
                let auto_bits = f32_bits(&ctx, &auto, elements);
                let auto_raw = fixed_explicit_vendor_raw_bytes(&ctx, &auto);
                fixed_ada_vendor_launch(
                    &ctx,
                    reference_ops,
                    shape,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
                );
                let reference_bits = f32_bits(&ctx, &reference, elements);
                let auto_error = fixed_ada_normalized_error(
                    &auto_bits,
                    &reference_bits,
                    custom_tolerance,
                    &format!("rung AUTO {row}/{} bias={has_bias}", cell.label),
                );
                fixed_ada_vendor_launch(&ctx, vendor_ops, shape, compute);
                let vendor_raw = fixed_explicit_vendor_raw_bytes(&ctx, &vendor);
                let vendor_error = fixed_ada_normalized_error(
                    &f32_bits(&ctx, &vendor, elements),
                    &reference_bits,
                    vendor_tolerance,
                    &format!(
                        "rung vendor {vendor_comparator} {row}/{} bias={has_bias}",
                        cell.label
                    ),
                );
                let auto_launch = || {
                    assert_eq!(
                        launch_fixed_auto_vendor_custom(&ctx, auto_ops, shape),
                        selected,
                        "explicit vendor AUTO tile changed during the rung survey"
                    );
                };
                let vendor_launch = || fixed_ada_vendor_launch(&ctx, vendor_ops, shape, compute);
                for &tile in &tiles {
                    let force_spec = fixed_force_spec(row, device.compute_capability, tile)
                        .unwrap_or_else(|error| panic!("invalid force inventory: {error}"));
                    assert_eq!(
                        (force_spec.input_dtype, force_spec.output_dtype),
                        (input_dtype, output_dtype)
                    );
                    match fixed_force_run_first_launch(&force_spec, has_bias, || {
                        inference_forward_with_tile(&ctx, forced_ops, shape, tile)
                    })
                    .unwrap_or_else(|error| panic!("{row}/{} {error}", cell.label))
                    {
                        FixedForceFirstLaunch::Runnable => {}
                        FixedForceFirstLaunch::BiasContractRejected { contract, reason } => {
                            println!(
                                concat!(
                                    "{{\"schema\":\"MambaBiFixedExplicitForcedRungRejectedV2\",{},\"row\":\"{}\",",
                                    "\"cell\":\"{}\",\"bias\":{},\"forced_tile\":\"{:?}\",",
                                    "\"gate\":\"bias_contract\",\"bias_contract\":\"{}\",\"reason\":\"{}\"}}"
                                ),
                                device_metadata,
                                row,
                                cell.label,
                                has_bias,
                                tile,
                                contract,
                                fixed_sm120_tf32_bd_json_escape(&reason)
                            );
                            rejected += 1;
                            continue;
                        }
                        FixedForceFirstLaunch::AvailabilityRejected(error) => {
                            println!(
                                concat!(
                                    "{{\"schema\":\"MambaBiFixedExplicitForcedRungRejectedV2\",{},\"row\":\"{}\",",
                                    "\"cell\":\"{}\",\"bias\":{},\"forced_tile\":\"{:?}\",",
                                    "\"gate\":\"launch_availability\",\"reason\":\"{}\"}}"
                                ),
                                device_metadata,
                                row,
                                cell.label,
                                has_bias,
                                tile,
                                fixed_sm120_tf32_bd_json_escape(&error)
                            );
                            rejected += 1;
                            continue;
                        }
                    }
                    let first = f32_bits(&ctx, &forced, elements);
                    let first_raw = fixed_explicit_vendor_raw_bytes(&ctx, &forced);
                    let label = format!("rung {row}/{} {tile:?} bias={has_bias}", cell.label);
                    let forced_error = fixed_ada_normalized_error(
                        &first,
                        &reference_bits,
                        custom_tolerance,
                        &label,
                    );
                    let forced_launch = || {
                        inference_forward_with_tile(&ctx, forced_ops, shape, tile)
                            .unwrap_or_else(|error| panic!("{label}: {error}"));
                    };
                    // Establish physical identity before timing or numeric-family
                    // rejection. RNA and both Ada half routes are mandatory even when
                    // graph timing is disabled: eager-only must not bypass ABI/argument gates.
                    let auto_identity_graph = if auto_descriptor.is_some()
                        || fixed_explicit_vendor_needs_identity_graph(&paths, selected)
                    {
                        Some(
                            unsafe {
                                capture_into_graph(&ctx.stream, || {
                                    auto_launch();
                                    Ok(())
                                })
                            }
                            .expect("capture AUTO physical-identity graph"),
                        )
                    } else {
                        None
                    };
                    let forced_identity_graph =
                        if fixed_explicit_vendor_needs_identity_graph(&paths, tile) {
                            let graph = unsafe {
                                capture_into_graph(&ctx.stream, || {
                                    forced_launch();
                                    Ok(())
                                })
                            }
                            .expect("capture forced physical-identity graph");
                            assert_eq!(
                                single_graph_kernel_name(&graph, &label),
                                force_spec.expected_symbol,
                                "{label} physical graph symbol"
                            );
                            Some(graph)
                        } else {
                            None
                        };
                    let auto_identity_inventory = (auto_descriptor.is_some()
                        || fixed_explicit_vendor_ada_descriptor(selected))
                    .then(|| {
                        fixed_explicit_vendor_graph_inventory(
                            auto_identity_graph.as_ref().unwrap_or_else(|| {
                                panic!("physical AUTO always captures an identity graph")
                            }),
                            "AUTO",
                            (auto_descriptor.is_none()
                                && fixed_explicit_vendor_ada_descriptor(selected))
                            .then_some((selected, input_dtype, auto_ops, shape)),
                            auto_descriptor.map(|descriptor| (descriptor, auto_ops, shape)),
                            None,
                        )
                    });
                    let forced_identity_inventory = (tile == InferenceTile::Tf32RnaM128N128S3
                        || fixed_explicit_vendor_ada_descriptor(tile))
                    .then(|| {
                        fixed_explicit_vendor_graph_inventory(
                            forced_identity_graph
                                .as_ref()
                                .expect("RNA/Ada half always captures an identity graph"),
                            if tile == InferenceTile::Tf32RnaM128N128S3 {
                                "Tf32RnaM128N128S3"
                            } else {
                                "forced"
                            },
                            fixed_explicit_vendor_ada_descriptor(tile).then_some((
                                tile,
                                input_dtype,
                                forced_ops,
                                shape,
                            )),
                            None,
                            None,
                        )
                    });
                    // A retune between M-dependent Fixed rungs is only eligible
                    // when it retains AUTO's per-element numeric family.
                    if first_raw != auto_raw {
                        println!(
                            concat!(
                                "{{\"schema\":\"MambaBiFixedExplicitForcedRungRejectedV2\",{},\"row\":\"{}\",",
                                "\"cell\":\"{}\",\"bias\":{},\"forced_tile\":\"{:?}\",",
                                "\"gate\":\"auto_bit_identity\",\"forced_normalized_error\":{}}}"
                            ),
                            device_metadata, row, cell.label, has_bias, tile, forced_error
                        );
                        rejected += 1;
                        continue;
                    }
                    forced_launch();
                    assert!(
                        fixed_explicit_vendor_raw_bytes(&ctx, &forced) == first_raw,
                        "{label} repeat bits differ"
                    );
                    for _ in 0..128 {
                        auto_launch();
                        forced_launch();
                        vendor_launch();
                    }
                    ctx.stream.synchronize().expect("rung warmup sync");
                    // Allocate and warm the complete workflows before capture.
                    // These graphs and all captured buffers outlive every replay.
                    let graphs = if paths.contains(&"graph") {
                        let auto_graph = auto_identity_graph
                            .expect("graph path must retain AUTO identity graph");
                        let forced_graph = forced_identity_graph
                            .expect("graph path must retain forced identity graph");
                        let vendor_graph = unsafe {
                            capture_into_graph(&ctx.stream, || {
                                vendor_launch();
                                Ok(())
                            })
                        }
                        .expect("capture complete vendor bias/GEMM workflow");
                        Some((auto_graph, forced_graph, vendor_graph))
                    } else {
                        None
                    };
                    let graph_inventory = if let Some((auto_graph, forced_graph, vendor_graph)) =
                        &graphs
                    {
                        let auto_inventory = fixed_explicit_vendor_graph_inventory(
                            auto_graph,
                            "AUTO",
                            (auto_descriptor.is_none()
                                && fixed_explicit_vendor_ada_descriptor(selected))
                            .then_some((selected, input_dtype, auto_ops, shape)),
                            auto_descriptor.map(|descriptor| (descriptor, auto_ops, shape)),
                            None,
                        );
                        let forced_inventory = fixed_explicit_vendor_graph_inventory(
                            forced_graph,
                            if tile == InferenceTile::Tf32Sm120M64S2PairStore {
                                "Tf32Sm120M64S2PairStore"
                            } else if tile == InferenceTile::Tf32RnaM128N128S3 {
                                "Tf32RnaM128N128S3"
                            } else {
                                "forced"
                            },
                            fixed_explicit_vendor_ada_descriptor(tile).then_some((
                                tile,
                                input_dtype,
                                forced_ops,
                                shape,
                            )),
                            None,
                            None,
                        );
                        let bias_symbol = has_bias.then_some(match output_dtype {
                            WeightDtype::F32 => "bias_broadcast",
                            WeightDtype::Bf16 => "bias_broadcast_bf16",
                            WeightDtype::F16 => "bias_broadcast_f16",
                        });
                        let vendor_inventory = fixed_explicit_vendor_graph_inventory(
                            vendor_graph,
                            "vendor",
                            None,
                            None,
                            bias_symbol,
                        );
                        format!(
                            "{{\"auto\":{auto_inventory},\"forced\":{forced_inventory},\"vendor\":{vendor_inventory}}}"
                        )
                    } else if auto_identity_inventory.is_some()
                        || forced_identity_inventory.is_some()
                    {
                        format!(
                            "{{\"auto\":{},\"forced\":{},\"vendor\":null}}",
                            auto_identity_inventory.as_deref().unwrap_or("null"),
                            forced_identity_inventory.as_deref().unwrap_or("null"),
                        )
                    } else {
                        "null".to_owned()
                    };
                    for &path in &paths {
                        let auto_run = || {
                            if path == "graph" {
                                graphs
                                    .as_ref()
                                    .unwrap()
                                    .0
                                    .launch()
                                    .expect("AUTO graph replay");
                            } else {
                                auto_launch();
                            }
                        };
                        let forced_run = || {
                            if path == "graph" {
                                graphs
                                    .as_ref()
                                    .unwrap()
                                    .1
                                    .launch()
                                    .expect("forced graph replay");
                            } else {
                                forced_launch();
                            }
                        };
                        let vendor_run = || {
                            if path == "graph" {
                                graphs
                                    .as_ref()
                                    .unwrap()
                                    .2
                                    .launch()
                                    .expect("vendor graph replay");
                            } else {
                                vendor_launch();
                            }
                        };
                        if path == "graph" {
                            for replay in 0..2 {
                                // A no-op graph must not pass on the old eager output.
                                for output in [&auto, &forced, &vendor] {
                                    fixed_explicit_vendor_poison_output(&ctx, output);
                                }
                                auto_run();
                                forced_run();
                                vendor_run();
                                assert!(
                                    fixed_explicit_vendor_raw_bytes(&ctx, &auto) == auto_raw,
                                    "{label} AUTO graph replay {replay} changed storage bits"
                                );
                                assert!(
                                    fixed_explicit_vendor_raw_bytes(&ctx, &forced) == first_raw,
                                    "{label} forced graph replay {replay} changed storage bits"
                                );
                                assert!(
                                    fixed_explicit_vendor_raw_bytes(&ctx, &vendor) == vendor_raw,
                                    "{label} vendor graph replay {replay} changed storage bits"
                                );
                            }
                            for _ in 0..128 {
                                auto_run();
                                forced_run();
                                vendor_run();
                            }
                        }
                        assert!(
                            fixed_explicit_vendor_raw_bytes(&ctx, &auto) == auto_raw,
                            "{label} {path} AUTO pre-timing storage bits differ"
                        );
                        assert!(
                            fixed_explicit_vendor_raw_bytes(&ctx, &forced) == first_raw,
                            "{label} {path} forced pre-timing storage bits differ"
                        );
                        assert!(
                            fixed_explicit_vendor_raw_bytes(&ctx, &vendor) == vendor_raw,
                            "{label} {path} vendor pre-timing storage bits differ"
                        );
                        let auto_iterations = fixed_auto_vendor_iterations(
                            fixed_ada_event_window_us(&ctx, 16, auto_run),
                        );
                        let forced_iterations = fixed_auto_vendor_iterations(
                            fixed_ada_event_window_us(&ctx, 16, forced_run),
                        );
                        let vendor_iterations = fixed_auto_vendor_iterations(
                            fixed_ada_event_window_us(&ctx, 16, vendor_run),
                        );
                        for auto_first in [true, false] {
                            let mut auto_samples = Vec::with_capacity(windows);
                            let mut forced_samples = Vec::with_capacity(windows);
                            let mut vendor_samples = Vec::with_capacity(windows);
                            let mut over_auto = Vec::with_capacity(windows);
                            let mut over_vendor = Vec::with_capacity(windows);
                            for _ in 0..windows {
                                let (auto_us, forced_us, vendor_us) = if auto_first {
                                    (
                                        fixed_ada_event_window_us(&ctx, auto_iterations, auto_run),
                                        fixed_ada_event_window_us(
                                            &ctx,
                                            forced_iterations,
                                            forced_run,
                                        ),
                                        fixed_ada_event_window_us(
                                            &ctx,
                                            vendor_iterations,
                                            vendor_run,
                                        ),
                                    )
                                } else {
                                    let vendor_us = fixed_ada_event_window_us(
                                        &ctx,
                                        vendor_iterations,
                                        vendor_run,
                                    );
                                    let forced_us = fixed_ada_event_window_us(
                                        &ctx,
                                        forced_iterations,
                                        forced_run,
                                    );
                                    (
                                        fixed_ada_event_window_us(&ctx, auto_iterations, auto_run),
                                        forced_us,
                                        vendor_us,
                                    )
                                };
                                auto_samples.push(auto_us);
                                forced_samples.push(forced_us);
                                vendor_samples.push(vendor_us);
                                over_auto.push(forced_us / auto_us);
                                over_vendor.push(forced_us / vendor_us);
                            }
                            assert!(
                                fixed_explicit_vendor_raw_bytes(&ctx, &forced) == first_raw,
                                "{label} {path} forced post-timing storage bits differ"
                            );
                            assert!(
                                fixed_explicit_vendor_raw_bytes(&ctx, &auto) == auto_raw,
                                "{label} {path} AUTO post-timing storage bits differ"
                            );
                            assert!(
                                fixed_explicit_vendor_raw_bytes(&ctx, &vendor) == vendor_raw,
                                "{label} {path} vendor post-timing storage bits differ"
                            );
                            let mut auto_sorted = auto_samples.clone();
                            let mut forced_sorted = forced_samples.clone();
                            let mut vendor_sorted = vendor_samples.clone();
                            auto_sorted.sort_by(f64::total_cmp);
                            forced_sorted.sort_by(f64::total_cmp);
                            vendor_sorted.sort_by(f64::total_cmp);
                            over_auto.sort_by(f64::total_cmp);
                            over_vendor.sort_by(f64::total_cmp);
                            println!(
                                concat!(
                                    "{{\"schema\":\"MambaBiFixedExplicitForcedRungV2\",{},\"row\":\"{}\",",
                                    "\"cell\":\"{}\",\"m\":{},\"k\":{},\"n\":{},",
                                    "\"tuning_table_revision\":{},\"dtype\":\"{}\",\"input_dtype\":\"{}\",\"output_dtype\":\"{}\",",
                                    "\"auto_tile\":\"{:?}\",\"forced_tile\":\"{:?}\",\"op\":\"nn\",\"path\":\"{}\",",
                                    "\"graphs\":{},\"graph_replay_bits_equal\":{},\"raw_storage_bits_equal\":true,\"vendor_repeat_bits_equal\":true,",
                                    "\"timing\":\"cuda_events\",\"alpha\":1,\"beta\":0,\"bias\":{},",
                                    "\"vendor_gemm_beta\":{},\"vendor_bias_broadcast_timed\":{},",
                                    "{},\"vendor_compute\":\"{:?}\",\"reference_compute\":\"CUBLAS_COMPUTE_32F_PEDANTIC\",",
                                    "\"reference_output_dtype\":\"f32\",\"auto_bits_equal\":true,\"repeat_bits_equal\":true,",
                                    "\"auto_normalized_error\":{},",
                                    "\"forced_normalized_error\":{},\"vendor_normalized_error\":{},\"order\":\"{}\",",
                                    "\"windows\":{},\"auto_iterations\":{},\"forced_iterations\":{},\"vendor_iterations\":{},",
                                    "\"auto_p50_us\":{},\"forced_p50_us\":{},\"vendor_p50_us\":{},",
                                    "\"forced_over_auto_p50\":{},\"forced_over_auto_p95\":{},",
                                    "\"forced_over_vendor_p50\":{},\"forced_over_vendor_p95\":{},",
                                    "\"auto_samples_us\":{:?},\"forced_samples_us\":{:?},\"vendor_samples_us\":{:?}}}"
                                ),
                                device_metadata,
                                row,
                                cell.label,
                                shape.m,
                                shape.k,
                                shape.n,
                                TUNING_TABLE_REVISION,
                                input_dtype.as_str(),
                                input_dtype.as_str(),
                                output_dtype.as_str(),
                                selected,
                                tile,
                                path,
                                graph_inventory,
                                path == "graph",
                                has_bias,
                                u8::from(has_bias),
                                has_bias,
                                fixed_explicit_vendor_tolerance_metadata(&row_spec),
                                compute,
                                auto_error,
                                forced_error,
                                vendor_error,
                                if auto_first {
                                    "auto_forced_vendor"
                                } else {
                                    "vendor_forced_auto"
                                },
                                windows,
                                auto_iterations,
                                forced_iterations,
                                vendor_iterations,
                                percentile(&auto_sorted, 0.5),
                                percentile(&forced_sorted, 0.5),
                                percentile(&vendor_sorted, 0.5),
                                percentile(&over_auto, 0.5),
                                percentile(&over_auto, 0.95),
                                percentile(&over_vendor, 0.5),
                                percentile(&over_vendor, 0.95),
                                auto_samples,
                                forced_samples,
                                vendor_samples
                            );
                            records += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(
        records > 0,
        "the explicit vendor rung survey measured no eligible candidates"
    );
    println!(
        "{{\"schema\":\"MambaBiFixedExplicitForcedRungCompleteV2\",{},{},\"records\":{records},\"rejected\":{rejected},\"passed\":true}}",
        device_metadata,
        fixed_exact_comparator_completion_metadata(),
    );
}

#[test]
#[ignore = "requires explicit Task8 post-AUTO44 controls and the admitted quiet CC8.9 Ada GPU"]
fn fixed_ada_exact_post_auto_paired_precision_cublas() {
    assert_eq!(
        std::env::var("MAMBA_FIXED_ADA_EXACT_POST_AUTO").as_deref(),
        Ok("1"),
        "MAMBA_FIXED_ADA_EXACT_POST_AUTO must be exactly 1"
    );
    toolkit_admission::run_post_auto();
}

fn fixed_ada_direct_pair_poison_complement(
    ctx: &GpuCtx,
    output: &DtypedBuf,
    expected: &[u8],
    label: &str,
) {
    assert_eq!(output.size_bytes(), expected.len(), "{label} poison length");
    let poison = expected.iter().map(|byte| !byte).collect::<Vec<_>>();
    assert_eq!(
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                output.cached_ptr(),
                poison.as_ptr().cast(),
                poison.len(),
                ctx.stream.cu_stream(),
            )
        },
        cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "{label} complement poison upload"
    );
    ctx.stream
        .synchronize()
        .unwrap_or_else(|error| panic!("{label} complement poison synchronization: {error}"));
}

#[test]
#[ignore = "requires MAMBA_FIXED_ADA_DIRECT_PAIR=1 and an exclusive quiet CC8.9 GPU; emits direct forced pipeline/swizzle evidence"]
fn fixed_ada_half_forced_direct_pair() {
    use cudarc::cublas::sys::cublasComputeType_t;

    assert_eq!(
        std::env::var("MAMBA_FIXED_ADA_DIRECT_PAIR").as_deref(),
        Ok("1"),
        "set MAMBA_FIXED_ADA_DIRECT_PAIR=1 to run direct forced pairing"
    );
    if cfg!(debug_assertions) {
        panic!("Ada half direct pairing requires --release");
    }
    let requested_tiles = match std::env::var("MAMBA_FIXED_VENDOR_TILES") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("read MAMBA_FIXED_VENDOR_TILES: {error}"),
    };
    fixed_ada_direct_pair_reject_vendor_tiles(requested_tiles.as_deref())
        .expect("direct-pair tile boundary");

    let all_rows = fixed_explicit_vendor_row_specs();
    let rows = [all_rows[0], all_rows[1]];
    let selected_rows = fixed_ada_filter("MAMBA_FIXED_ADA_ROWS", &["bf16", "f16"]);
    let labels = FIXED_AUTO_VENDOR_EXACT_CELLS
        .iter()
        .map(|cell| cell.label)
        .collect::<Vec<_>>();
    let selected_cells = fixed_ada_filter("MAMBA_FIXED_ADA_CELLS", &labels);
    let biases = fixed_ada_filter("MAMBA_FIXED_ADA_BIAS", &["0", "1"]);
    let requested_paths = match std::env::var("MAMBA_FIXED_VENDOR_PATHS") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("read MAMBA_FIXED_VENDOR_PATHS: {error}"),
    };
    let paths =
        fixed_explicit_vendor_paths(requested_paths.as_deref()).expect("direct-pair path filter");
    let windows = match std::env::var("MAMBA_FIXED_ADA_WINDOWS") {
        Ok(value) => value
            .parse::<usize>()
            .expect("MAMBA_FIXED_ADA_WINDOWS must be an integer"),
        Err(std::env::VarError::NotPresent) => 101,
        Err(error) => panic!("read MAMBA_FIXED_ADA_WINDOWS: {error}"),
    };
    assert!(
        (1..=10_001).contains(&windows),
        "direct-pair windows must be in 1..=10001"
    );

    fixed_sm120_tf32_bd_environment_preflight("Ada half direct forced pipeline/swizzle")
        .expect("Ada half direct-pair environment preflight");
    let device = GpuDevice::new(0).expect("Ada half direct-pair CUDA device");
    let requested_cc = match std::env::var("MAMBA_FIXED_VENDOR_EXACT_CC") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("read MAMBA_FIXED_VENDOR_EXACT_CC: {error}"),
    };
    fixed_explicit_vendor_admit_cc(requested_cc.as_deref(), device.compute_capability)
        .expect("Ada half direct-pair exact-CC admission");
    assert_eq!(device.compute_capability, (8, 9), "direct-pair CC");
    assert_eq!(device.multiprocessor_count(), 142, "direct-pair SM count");
    let ctx = GpuCtx::new(&device).expect("Ada half direct-pair GPU context");
    let compiler = ctx.kernels.compiler_identity();
    assert!(
        compiler.nvrtc_library_known,
        "direct pairing requires a known NVRTC library"
    );
    assert!(
        matches!(compiler.nvrtc_version, (12, 8) | (13, 0) | (13, 2)),
        "unsupported direct-pair NVRTC {:?}",
        compiler.nvrtc_version
    );
    let fixed_artifact = ctx.kernels.artifact_set_identity().fixed;
    let device_metadata = format!(
        concat!(
            "\"cc\":\"{}.{}\",\"sm_count\":{},\"nvrtc\":[{},{}],",
            "\"compiler_target\":\"{:?}\",\"fixed_source_digest\":\"{}\",",
            "\"fixed_invocation_digest\":\"{}\",\"fixed_artifact_digest\":\"{}\",",
            "\"header_manifest_digest\":\"{}\",\"nvrtc_library_domain\":\"{}\",",
            "\"nvrtc_library_known\":true"
        ),
        device.compute_capability.0,
        device.compute_capability.1,
        device.multiprocessor_count(),
        compiler.nvrtc_version.0,
        compiler.nvrtc_version.1,
        compiler.target,
        digest_hex(&compiler.source_digest),
        digest_hex(&compiler.invocation_digest),
        digest_hex(&fixed_artifact.artifact_digest),
        digest_hex(&compiler.header_manifest_digest),
        digest_hex(&compiler.nvrtc_library_domain),
    );

    let pipeline_tile = InferenceTile::Tc128Sm89Pipeline;
    let swizzle_tile = InferenceTile::Tc128Sm89Swizzle;
    let mut records = 0usize;
    for row_index in selected_rows {
        let row_spec = rows[row_index];
        let row = row_spec.name;
        let input_dtype = row_spec.input_dtype;
        let output_dtype = row_spec.output_dtype;
        assert_eq!(input_dtype, output_dtype, "direct-pair homogeneous row");
        configure_fixed_auto_vendor_custom(&ctx, row_spec.policy);
        for &cell_index in &selected_cells {
            let cell = FIXED_AUTO_VENDOR_EXACT_CELLS[cell_index];
            let shape = cell.shape;
            let elements = shape.m * shape.n;
            let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, input_dtype)
                .expect("direct-pair A");
            let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, input_dtype)
                .expect("direct-pair B");
            a.upload_f32(&ctx.stream, &synth(shape.m * shape.k, 0x0ada_a001))
                .expect("direct-pair A upload");
            b.upload_f32(&ctx.stream, &synth(shape.k * shape.n, 0x0ada_b001))
                .expect("direct-pair B upload");
            let auto =
                DtypedBuf::zeros(&ctx.stream, elements, output_dtype).expect("direct-pair AUTO C");
            let pipeline = DtypedBuf::zeros(&ctx.stream, elements, output_dtype)
                .expect("direct-pair pipeline C");
            let swizzle = DtypedBuf::zeros(&ctx.stream, elements, output_dtype)
                .expect("direct-pair swizzle C");
            let reference = DtypedBuf::zeros(&ctx.stream, elements, WeightDtype::F32)
                .expect("direct-pair PEDANTIC reference C");
            let bias = DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32)
                .expect("direct-pair F32 bias");
            bias.upload_f32(&ctx.stream, &synth(shape.n, 0x0ada_b1a5))
                .expect("direct-pair F32 bias upload");

            for &bias_index in &biases {
                let has_bias = bias_index == 1;
                let expected_auto =
                    expected_ada_half_auto(compiler.nvrtc_version, input_dtype, shape, has_bias)
                        .expect("literal revision-45 direct-pair AUTO expectation");
                let auto_ops = InferenceFwdOperands {
                    c: typed(&auto, output_dtype),
                    x: typed(&a, input_dtype),
                    w: typed(&b, input_dtype),
                    bias_ptr: has_bias.then(|| bias.cached_ptr()),
                };
                let pipeline_ops = InferenceFwdOperands {
                    c: typed(&pipeline, output_dtype),
                    ..auto_ops
                };
                let swizzle_ops = InferenceFwdOperands {
                    c: typed(&swizzle, output_dtype),
                    ..auto_ops
                };
                let reference_ops = InferenceFwdOperands {
                    c: typed(&reference, WeightDtype::F32),
                    ..auto_ops
                };

                let actual_auto = launch_fixed_auto_vendor_custom(&ctx, auto_ops, shape);
                assert_eq!(
                    actual_auto, expected_auto,
                    "direct-pair actual AUTO changed for {row}/{} bias={has_bias}",
                    cell.label
                );
                let auto_raw = fixed_explicit_vendor_raw_bytes(&ctx, &auto);
                let auto_bits = f32_bits(&ctx, &auto, elements);
                fixed_ada_vendor_launch(
                    &ctx,
                    reference_ops,
                    shape,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
                );
                let reference_bits = f32_bits(&ctx, &reference, elements);
                let auto_error = fixed_ada_normalized_error(
                    &auto_bits,
                    &reference_bits,
                    row_spec.custom_tolerance,
                    &format!("direct-pair AUTO {row}/{} bias={has_bias}", cell.label),
                );

                for tile in [pipeline_tile, swizzle_tile] {
                    let spec = fixed_force_spec(row, device.compute_capability, tile)
                        .unwrap_or_else(|error| panic!("direct-pair force spec: {error}"));
                    assert_eq!(
                        (spec.input_dtype, spec.output_dtype),
                        (input_dtype, output_dtype)
                    );
                }
                let pipeline_launch = || {
                    inference_forward_with_tile(&ctx, pipeline_ops, shape, pipeline_tile)
                        .unwrap_or_else(|error| panic!("direct-pair pipeline launch: {error}"));
                };
                let swizzle_launch = || {
                    inference_forward_with_tile(&ctx, swizzle_ops, shape, swizzle_tile)
                        .unwrap_or_else(|error| panic!("direct-pair swizzle launch: {error}"));
                };
                pipeline_launch();
                let pipeline_raw = fixed_explicit_vendor_raw_bytes(&ctx, &pipeline);
                let pipeline_bits = f32_bits(&ctx, &pipeline, elements);
                swizzle_launch();
                let swizzle_raw = fixed_explicit_vendor_raw_bytes(&ctx, &swizzle);
                let swizzle_bits = f32_bits(&ctx, &swizzle, elements);
                assert_eq!(pipeline_raw, auto_raw, "direct-pair pipeline/AUTO raw bits");
                assert_eq!(swizzle_raw, auto_raw, "direct-pair swizzle/AUTO raw bits");
                let pipeline_error = fixed_ada_normalized_error(
                    &pipeline_bits,
                    &reference_bits,
                    row_spec.custom_tolerance,
                    &format!("direct-pair pipeline {row}/{} bias={has_bias}", cell.label),
                );
                let swizzle_error = fixed_ada_normalized_error(
                    &swizzle_bits,
                    &reference_bits,
                    row_spec.custom_tolerance,
                    &format!("direct-pair swizzle {row}/{} bias={has_bias}", cell.label),
                );

                fixed_ada_direct_pair_poison_complement(
                    &ctx,
                    &pipeline,
                    &pipeline_raw,
                    "direct-pair pipeline repeat",
                );
                pipeline_launch();
                assert_eq!(
                    fixed_explicit_vendor_raw_bytes(&ctx, &pipeline),
                    pipeline_raw,
                    "direct-pair pipeline repeat bits"
                );
                fixed_ada_direct_pair_poison_complement(
                    &ctx,
                    &swizzle,
                    &swizzle_raw,
                    "direct-pair swizzle repeat",
                );
                swizzle_launch();
                assert_eq!(
                    fixed_explicit_vendor_raw_bytes(&ctx, &swizzle),
                    swizzle_raw,
                    "direct-pair swizzle repeat bits"
                );

                let pipeline_graph = unsafe {
                    capture_into_graph(&ctx.stream, || {
                        pipeline_launch();
                        Ok(())
                    })
                }
                .expect("capture direct-pair pipeline graph");
                let swizzle_graph = unsafe {
                    capture_into_graph(&ctx.stream, || {
                        swizzle_launch();
                        Ok(())
                    })
                }
                .expect("capture direct-pair swizzle graph");
                let pipeline_inventory = fixed_explicit_vendor_graph_inventory(
                    &pipeline_graph,
                    "pipeline",
                    Some((pipeline_tile, input_dtype, pipeline_ops, shape)),
                    None,
                    None,
                );
                let swizzle_inventory = fixed_explicit_vendor_graph_inventory(
                    &swizzle_graph,
                    "swizzle",
                    Some((swizzle_tile, input_dtype, swizzle_ops, shape)),
                    None,
                    None,
                );
                let graph_inventory = format!(
                    "{{\"pipeline\":{pipeline_inventory},\"swizzle\":{swizzle_inventory}}}"
                );

                for &path in &paths {
                    let pipeline_run = || {
                        if path == "graph" {
                            pipeline_graph
                                .launch()
                                .expect("direct-pair pipeline graph replay");
                        } else {
                            pipeline_launch();
                        }
                    };
                    let swizzle_run = || {
                        if path == "graph" {
                            swizzle_graph
                                .launch()
                                .expect("direct-pair swizzle graph replay");
                        } else {
                            swizzle_launch();
                        }
                    };

                    let replay_checks = if path == "graph" { 2 } else { 1 };
                    for replay in 0..replay_checks {
                        fixed_ada_direct_pair_poison_complement(
                            &ctx,
                            &pipeline,
                            &pipeline_raw,
                            "direct-pair pipeline path check",
                        );
                        pipeline_run();
                        assert_eq!(
                            fixed_explicit_vendor_raw_bytes(&ctx, &pipeline),
                            pipeline_raw,
                            "direct-pair pipeline {path} replay {replay} bits"
                        );
                        fixed_ada_direct_pair_poison_complement(
                            &ctx,
                            &swizzle,
                            &swizzle_raw,
                            "direct-pair swizzle path check",
                        );
                        swizzle_run();
                        assert_eq!(
                            fixed_explicit_vendor_raw_bytes(&ctx, &swizzle),
                            swizzle_raw,
                            "direct-pair swizzle {path} replay {replay} bits"
                        );
                    }
                    for _ in 0..128 {
                        pipeline_run();
                        swizzle_run();
                    }
                    ctx.stream.synchronize().expect("direct-pair warmup sync");
                    let pipeline_iterations = fixed_auto_vendor_iterations(
                        fixed_ada_event_window_us(&ctx, 16, pipeline_run),
                    );
                    let swizzle_iterations = fixed_auto_vendor_iterations(
                        fixed_ada_event_window_us(&ctx, 16, swizzle_run),
                    );

                    for order in ["pipeline_swizzle", "swizzle_pipeline"] {
                        fixed_ada_direct_pair_poison_complement(
                            &ctx,
                            &pipeline,
                            &pipeline_raw,
                            "direct-pair pipeline pre-order",
                        );
                        pipeline_run();
                        assert_eq!(
                            fixed_explicit_vendor_raw_bytes(&ctx, &pipeline),
                            pipeline_raw,
                            "direct-pair pipeline {path}/{order} pre-timing bits"
                        );
                        fixed_ada_direct_pair_poison_complement(
                            &ctx,
                            &swizzle,
                            &swizzle_raw,
                            "direct-pair swizzle pre-order",
                        );
                        swizzle_run();
                        assert_eq!(
                            fixed_explicit_vendor_raw_bytes(&ctx, &swizzle),
                            swizzle_raw,
                            "direct-pair swizzle {path}/{order} pre-timing bits"
                        );

                        let mut pipeline_samples = Vec::with_capacity(windows);
                        let mut swizzle_samples = Vec::with_capacity(windows);
                        for _ in 0..windows {
                            let (pipeline_us, swizzle_us) = fixed_ada_direct_pair_ordered_window(
                                order,
                                || {
                                    fixed_ada_event_window_us(
                                        &ctx,
                                        pipeline_iterations,
                                        pipeline_run,
                                    )
                                },
                                || fixed_ada_event_window_us(&ctx, swizzle_iterations, swizzle_run),
                            )
                            .expect("known direct-pair order");
                            pipeline_samples.push(pipeline_us);
                            swizzle_samples.push(swizzle_us);
                        }
                        assert_eq!(
                            fixed_explicit_vendor_raw_bytes(&ctx, &pipeline),
                            pipeline_raw,
                            "direct-pair pipeline {path}/{order} post-timing bits"
                        );
                        assert_eq!(
                            fixed_explicit_vendor_raw_bytes(&ctx, &swizzle),
                            swizzle_raw,
                            "direct-pair swizzle {path}/{order} post-timing bits"
                        );
                        let mut pipeline_sorted = pipeline_samples.clone();
                        let mut swizzle_sorted = swizzle_samples.clone();
                        pipeline_sorted.sort_by(f64::total_cmp);
                        swizzle_sorted.sort_by(f64::total_cmp);
                        let ((swizzle_p50, swizzle_p95), (pipeline_p50, pipeline_p95)) =
                            fixed_ada_direct_pair_ratio_quantiles(
                                &pipeline_samples,
                                &swizzle_samples,
                            );
                        println!(
                            concat!(
                                "{{\"schema\":\"MambaBiFixedHalfDirectPairV1\",{},",
                                "\"tuning_table_revision\":{},\"row\":\"{}\",\"cell\":\"{}\",",
                                "\"m\":{},\"k\":{},\"n\":{},\"dtype\":\"{}\",",
                                "\"input_dtype\":\"{}\",\"output_dtype\":\"{}\",\"op\":\"nn\",",
                                "\"path\":\"{}\",\"bias\":{},\"alpha\":1,\"beta\":0,",
                                "\"timing\":\"cuda_events\",\"pipeline_tile\":\"{:?}\",",
                                "\"swizzle_tile\":\"{:?}\",\"graphs\":{},",
                                "\"actual_auto_tile\":\"{:?}\",\"auto_bits_equal\":true,",
                                "\"raw_storage_bits_equal\":true,\"repeat_bits_equal\":true,",
                                "\"graph_replay_bits_equal\":{},",
                                "\"reference_compute\":\"CUBLAS_COMPUTE_32F_PEDANTIC\",",
                                "\"reference_output_dtype\":\"f32\",",
                                "\"custom_normalized_error_tolerance\":{},",
                                "\"auto_normalized_error\":{},\"pipeline_normalized_error\":{},",
                                "\"swizzle_normalized_error\":{},\"order\":\"{}\",\"windows\":{},",
                                "\"pipeline_iterations\":{},\"swizzle_iterations\":{},",
                                "\"pipeline_p50_us\":{},\"swizzle_p50_us\":{},",
                                "\"pipeline_samples_us\":{:?},\"swizzle_samples_us\":{:?},",
                                "\"swizzle_over_pipeline_p50\":{},\"swizzle_over_pipeline_p95\":{},",
                                "\"pipeline_over_swizzle_p50\":{},\"pipeline_over_swizzle_p95\":{}}}"
                            ),
                            device_metadata,
                            TUNING_TABLE_REVISION,
                            row,
                            cell.label,
                            shape.m,
                            shape.k,
                            shape.n,
                            input_dtype.as_str(),
                            input_dtype.as_str(),
                            output_dtype.as_str(),
                            path,
                            has_bias,
                            pipeline_tile,
                            swizzle_tile,
                            graph_inventory,
                            actual_auto,
                            path == "graph",
                            row_spec.custom_tolerance,
                            auto_error,
                            pipeline_error,
                            swizzle_error,
                            order,
                            windows,
                            pipeline_iterations,
                            swizzle_iterations,
                            percentile(&pipeline_sorted, 0.5),
                            percentile(&swizzle_sorted, 0.5),
                            pipeline_samples,
                            swizzle_samples,
                            swizzle_p50,
                            swizzle_p95,
                            pipeline_p50,
                            pipeline_p95,
                        );
                        records += 1;
                    }
                }
            }
        }
    }
    assert!(
        records > 0,
        "direct pairing measured no selected route sets"
    );
    println!(
        "{{\"schema\":\"MambaBiFixedHalfDirectPairCompleteV1\",{},\"tuning_table_revision\":{},\"records\":{records},\"rejected\":0,\"passed\":true}}",
        device_metadata, TUNING_TABLE_REVISION,
    );
}

const F32_N128_GUARD: usize = 32;
const F32_N128_SENTINEL: f32 = 19.25;

fn f32_n128_offset_buffer(ctx: &GpuCtx, logical: &[f32], offset: usize, label: &str) -> DtypedBuf {
    let storage_len = offset + logical.len().max(1);
    let mut storage = vec![-31.5f32; storage_len];
    if !logical.is_empty() {
        storage[offset..offset + logical.len()].copy_from_slice(logical);
    }
    let buffer = DtypedBuf::zeros(&ctx.stream, storage_len, WeightDtype::F32)
        .unwrap_or_else(|error| panic!("{label} allocation: {error}"));
    buffer
        .upload_f32(&ctx.stream, &storage)
        .unwrap_or_else(|error| panic!("{label} upload: {error}"));
    buffer
}

fn f32_n128_guarded_output(
    ctx: &GpuCtx,
    output_len: usize,
    output_offset: usize,
    poison: f32,
    label: &str,
) -> DtypedBuf {
    let storage_len = F32_N128_GUARD + output_offset + output_len + F32_N128_GUARD;
    let mut storage = vec![F32_N128_SENTINEL; storage_len];
    let start = F32_N128_GUARD + output_offset;
    storage[start..start + output_len].fill(poison);
    let output = DtypedBuf::zeros(&ctx.stream, storage_len, WeightDtype::F32)
        .unwrap_or_else(|error| panic!("{label} allocation: {error}"));
    output
        .upload_f32(&ctx.stream, &storage)
        .unwrap_or_else(|error| panic!("{label} initialization: {error}"));
    output
}

fn f32_n128_output_operands(
    output: &DtypedBuf,
    output_offset: usize,
    a: &DtypedBuf,
    b: &DtypedBuf,
    input_offset: usize,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
) -> InferenceFwdOperands {
    let element_bytes = WeightDtype::F32.size_bytes() as u64;
    InferenceFwdOperands {
        c: TypedPtr {
            ptr: output.cached_ptr() + ((F32_N128_GUARD + output_offset) as u64 * element_bytes),
            dtype: WeightDtype::F32,
        },
        x: TypedPtr {
            ptr: a.cached_ptr() + (input_offset as u64 * element_bytes),
            dtype: WeightDtype::F32,
        },
        w: TypedPtr {
            ptr: b.cached_ptr() + (input_offset as u64 * element_bytes),
            dtype: WeightDtype::F32,
        },
        bias_ptr,
    }
}

fn f32_n128_logical_and_guards(
    ctx: &GpuCtx,
    output: &DtypedBuf,
    output_len: usize,
    output_offset: usize,
    label: &str,
) -> Vec<u32> {
    let all = f32_bits(
        ctx,
        output,
        F32_N128_GUARD + output_offset + output_len + F32_N128_GUARD,
    );
    let start = F32_N128_GUARD + output_offset;
    let sentinel = F32_N128_SENTINEL.to_bits();
    assert!(
        all[..start].iter().all(|&bits| bits == sentinel),
        "{label} leading guard changed"
    );
    assert!(
        all[start + output_len..]
            .iter()
            .all(|&bits| bits == sentinel),
        "{label} trailing guard changed"
    );
    all[start..start + output_len].to_vec()
}

#[test]
#[ignore = "requires an SM80+ CUDA device"]
fn fixed_f32_n128_matches_legacy_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let measured_device =
        device.compute_capability == (12, 0) && device.multiprocessor_count() == 170;
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let compiler = ctx.kernels.compiler_identity();
    let measured_stack =
        measured_device && compiler.nvrtc_version == (13, 2) && compiler.nvrtc_library_known;
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFma);

    let shapes = [
        InferenceShape { m: 1, k: 0, n: 1 },
        InferenceShape {
            m: 63,
            k: 31,
            n: 127,
        },
        InferenceShape {
            m: 64,
            k: 32,
            n: 128,
        },
        InferenceShape {
            m: 65,
            k: 33,
            n: 129,
        },
        InferenceShape {
            m: 63,
            k: 63,
            n: 129,
        },
        InferenceShape {
            m: 64,
            k: 64,
            n: 128,
        },
        InferenceShape {
            m: 65,
            k: 65,
            n: 127,
        },
        InferenceShape {
            m: 4621,
            k: 384,
            n: 1928,
        },
        InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 384,
        },
        InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
    ];

    for shape in shapes {
        let a_host = synth(shape.m * shape.k, 0xa128_0001 ^ shape.k as u64);
        let b_host = synth(shape.k * shape.n, 0xb128_0001 ^ shape.n as u64);
        let bias_host = synth(shape.n, 0xb1a5_1280 ^ shape.m as u64);
        let bias = f32_n128_offset_buffer(&ctx, &bias_host, 0, "bias");
        let output_len = shape.m * shape.n;
        for input_offset in [0usize, 1] {
            let a = f32_n128_offset_buffer(&ctx, &a_host, input_offset, "A");
            let b = f32_n128_offset_buffer(&ctx, &b_host, input_offset, "B");
            for output_offset in [0usize, 1] {
                for bias_ptr in [None, Some(bias.cached_ptr())] {
                    let oracle = f32_n128_guarded_output(
                        &ctx,
                        output_len,
                        output_offset,
                        -3.0,
                        "legacy oracle",
                    );
                    let auto = f32_n128_guarded_output(
                        &ctx,
                        output_len,
                        output_offset,
                        5.0,
                        "production AUTO",
                    );
                    let legacy = f32_n128_guarded_output(
                        &ctx,
                        output_len,
                        output_offset,
                        7.0,
                        "incumbent S2",
                    );
                    let candidate_a = f32_n128_guarded_output(
                        &ctx,
                        output_len,
                        output_offset,
                        -7.0,
                        "candidate A",
                    );
                    let candidate_b = f32_n128_guarded_output(
                        &ctx,
                        output_len,
                        output_offset,
                        9.0,
                        "candidate B",
                    );
                    let oracle_ops = f32_n128_output_operands(
                        &oracle,
                        output_offset,
                        &a,
                        &b,
                        input_offset,
                        bias_ptr,
                    );
                    let auto_ops = InferenceFwdOperands {
                        c: f32_n128_output_operands(
                            &auto,
                            output_offset,
                            &a,
                            &b,
                            input_offset,
                            bias_ptr,
                        )
                        .c,
                        ..oracle_ops
                    };
                    let legacy_ops = InferenceFwdOperands {
                        c: f32_n128_output_operands(
                            &legacy,
                            output_offset,
                            &a,
                            &b,
                            input_offset,
                            bias_ptr,
                        )
                        .c,
                        ..oracle_ops
                    };
                    let candidate_a_ops = InferenceFwdOperands {
                        c: f32_n128_output_operands(
                            &candidate_a,
                            output_offset,
                            &a,
                            &b,
                            input_offset,
                            bias_ptr,
                        )
                        .c,
                        ..oracle_ops
                    };
                    let candidate_b_ops = InferenceFwdOperands {
                        c: f32_n128_output_operands(
                            &candidate_b,
                            output_offset,
                            &a,
                            &b,
                            input_offset,
                            bias_ptr,
                        )
                        .c,
                        ..oracle_ops
                    };

                    inference_forward_f32_legacy_baseline(&ctx, oracle_ops, shape)
                        .expect("legacy oracle launch");
                    inference_forward_with_tile(&ctx, legacy_ops, shape, InferenceTile::Legacy)
                        .expect("forced incumbent S2 launch");
                    let selected = inference_forward(
                        &ctx,
                        auto_ops.c,
                        auto_ops.x,
                        auto_ops.w,
                        bias_ptr,
                        (shape.m, shape.k, shape.n),
                    )
                    .expect("production S2 launch");
                    let dims = (shape.m, shape.k, shape.n);
                    let specialized_alignment = input_offset == 0 && output_offset == 0;
                    let expected_auto = match (dims, bias_ptr.is_some()) {
                        ((4621, 384, 1928), false) if measured_stack && specialized_alignment => {
                            InferenceTile::F32Sm120TmaFmaM64N128
                        }
                        ((4621, 384, 1928), true) if measured_stack && specialized_alignment => {
                            if ctx
                                .kernels
                                .fixed_sm120_fma_postbias
                                .as_ref()
                                .is_some_and(|kernels| kernels.m128n64_t256.is_some())
                            {
                                InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64T256
                            } else {
                                InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64
                            }
                        }
                        ((4621, 768, 2304), false) if measured_stack && specialized_alignment => {
                            InferenceTile::F32Sm120TmaFmaM128N64
                        }
                        ((4621, 768, 2304), true) if measured_stack && specialized_alignment => {
                            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N64
                        }
                        ((4621, 1928, 384), true) if measured_stack && specialized_alignment => {
                            InferenceTile::F32Sm120TmaFmaFixedPostBiasM128N96
                        }
                        ((2048, 768, 2304), _) if measured_stack && specialized_alignment => {
                            InferenceTile::F32Sm120N64CopyPlan
                        }
                        ((2048, 2304, 768), _) if measured_stack && specialized_alignment => {
                            InferenceTile::F32Sm120N64CopyPlan
                        }
                        ((4621, 384, 1928) | (4621, 768, 2304), _) if measured_device => {
                            InferenceTile::F32N128S2
                        }
                        _ => InferenceTile::Legacy,
                    };
                    assert_eq!(selected, expected_auto, "production exact-F32 AUTO route");
                    if measured_stack
                        && specialized_alignment
                        && bias_ptr.is_some()
                        && matches!(dims, (4621, 768, 2304) | (4621, 1928, 384))
                    {
                        let graph = unsafe {
                            capture_into_graph(&ctx.stream, || {
                                inference_forward(
                                    &ctx, auto_ops.c, auto_ops.x, auto_ops.w, bias_ptr, dims,
                                )
                                .map(|_| ())
                            })
                        }
                        .expect("capture promoted B1/C1 AUTO graph");
                        let symbol = if dims == (4621, 768, 2304) {
                            "nn_sm120_tma_fma_postbias_m128n64_bk16_s2"
                        } else {
                            "nn_sm120_tma_fma_postbias_m128n96_bk16_s2"
                        };
                        assert_eq!(single_graph_kernel_name(&graph, "B1/C1 AUTO"), symbol);
                        assert_sm120_exact_tma_graph_contract(
                            &graph,
                            auto_ops.c.ptr,
                            shape,
                            bias_ptr,
                            expected_auto,
                        );
                        graph.launch().expect("replay promoted B1/C1 AUTO graph");
                    }
                    inference_forward_with_tile(
                        &ctx,
                        candidate_a_ops,
                        shape,
                        InferenceTile::F32N128S2,
                    )
                    .expect("first forced N128 launch");
                    inference_forward_with_tile(
                        &ctx,
                        candidate_b_ops,
                        shape,
                        InferenceTile::F32N128S2,
                    )
                    .expect("second forced N128 launch");
                    ctx.stream.synchronize().expect("N128 bit-gate sync");

                    let label = format!(
                        "M{} K{} N{} bias={} input_offset={} output_offset={}",
                        shape.m,
                        shape.k,
                        shape.n,
                        bias_ptr.is_some(),
                        input_offset,
                        output_offset
                    );
                    let oracle_bits = f32_n128_logical_and_guards(
                        &ctx,
                        &oracle,
                        output_len,
                        output_offset,
                        &format!("oracle {label}"),
                    );
                    let auto_bits = f32_n128_logical_and_guards(
                        &ctx,
                        &auto,
                        output_len,
                        output_offset,
                        &format!("AUTO {label}"),
                    );
                    let legacy_bits = f32_n128_logical_and_guards(
                        &ctx,
                        &legacy,
                        output_len,
                        output_offset,
                        &format!("incumbent S2 {label}"),
                    );
                    let candidate_a_bits = f32_n128_logical_and_guards(
                        &ctx,
                        &candidate_a,
                        output_len,
                        output_offset,
                        &format!("candidate A {label}"),
                    );
                    let candidate_b_bits = f32_n128_logical_and_guards(
                        &ctx,
                        &candidate_b,
                        output_len,
                        output_offset,
                        &format!("candidate B {label}"),
                    );
                    assert_eq!(candidate_a_bits, oracle_bits, "candidate/oracle {label}");
                    assert_eq!(
                        candidate_a_bits, legacy_bits,
                        "candidate/incumbent S2 {label}"
                    );
                    assert_eq!(auto_bits, oracle_bits, "AUTO/oracle {label}");
                    assert_eq!(candidate_a_bits, candidate_b_bits, "repeatability {label}");
                }
            }
        }
    }

    let shape = InferenceShape {
        m: 65,
        k: 65,
        n: 127,
    };
    let a_host = synth(shape.m * shape.k, 0xa128_0a11);
    let b_host = synth(shape.k * shape.n, 0xb128_0b11);
    let bias_host = synth(shape.n, 0xb128_b1a5);
    let a = f32_n128_offset_buffer(&ctx, &a_host, 1, "graph A");
    let b = f32_n128_offset_buffer(&ctx, &b_host, 1, "graph B");
    let bias = f32_n128_offset_buffer(&ctx, &bias_host, 0, "graph bias");
    let output = f32_n128_guarded_output(&ctx, shape.m * shape.n, 1, -13.0, "graph output");
    let operands = f32_n128_output_operands(&output, 1, &a, &b, 1, Some(bias.cached_ptr()));
    let run = || inference_forward_with_tile(&ctx, operands, shape, InferenceTile::F32N128S2);
    run().expect("eager N128 launch");
    ctx.stream.synchronize().expect("eager N128 sync");
    let eager = f32_n128_logical_and_guards(&ctx, &output, shape.m * shape.n, 1, "eager");
    let graph = unsafe { capture_into_graph(&ctx.stream, run) }.expect("capture N128 graph");
    for replay in 0..10 {
        graph.launch().expect("N128 graph launch");
        ctx.stream.synchronize().expect("N128 graph sync");
        assert_eq!(
            f32_n128_logical_and_guards(&ctx, &output, shape.m * shape.n, 1, "graph"),
            eager,
            "N128 graph replay {replay} changed bits"
        );
    }

    let k = 65usize;
    let n = 127usize;
    let large_a_host = synth(65 * k, 0xa128_0065);
    let b_host = synth(k * n, 0xb128_0065);
    let bias_host = synth(n, 0xb1a5_0065);
    let small_a = f32_n128_offset_buffer(&ctx, &large_a_host[..64 * k], 0, "prefix A64");
    let large_a = f32_n128_offset_buffer(&ctx, &large_a_host, 0, "prefix A65");
    let b = f32_n128_offset_buffer(&ctx, &b_host, 0, "prefix B");
    let bias = f32_n128_offset_buffer(&ctx, &bias_host, 0, "prefix bias");
    let small = f32_n128_guarded_output(&ctx, 64 * n, 0, -17.0, "prefix C64");
    let large = f32_n128_guarded_output(&ctx, 65 * n, 0, 23.0, "prefix C65");
    let small_shape = InferenceShape { m: 64, k, n };
    let large_shape = InferenceShape { m: 65, k, n };
    inference_forward_with_tile(
        &ctx,
        f32_n128_output_operands(&small, 0, &small_a, &b, 0, Some(bias.cached_ptr())),
        small_shape,
        InferenceTile::F32N128S2,
    )
    .expect("M64 prefix launch");
    inference_forward_with_tile(
        &ctx,
        f32_n128_output_operands(&large, 0, &large_a, &b, 0, Some(bias.cached_ptr())),
        large_shape,
        InferenceTile::F32N128S2,
    )
    .expect("M65 prefix launch");
    ctx.stream.synchronize().expect("prefix sync");
    assert_eq!(
        f32_n128_logical_and_guards(&ctx, &small, 64 * n, 0, "M64 prefix"),
        f32_n128_logical_and_guards(&ctx, &large, 65 * n, 0, "M65 prefix")[..64 * n],
        "N128 first 64 rows changed when M grew from 64 to 65"
    );
}

#[test]
#[ignore = "requires an SM120 CUDA device"]
fn fixed_f32_n128_resource_gate_sm120() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(
        device.compute_capability,
        (12, 0),
        "resource gate requires CC12.0"
    );
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert_fixed_f32_n128_resources(&ctx);
}

fn assert_fixed_f32_n128_resources(ctx: &GpuCtx) {
    let function = &ctx.kernels.gemm_bi_f32_f32_n128_s2;
    let static_shared_bytes = function
        .shared_size_bytes()
        .expect("N128 static shared size");
    let local_bytes = function.local_size_bytes().expect("N128 local size");
    let registers = function.num_regs().expect("N128 registers");
    let occupancy = function
        .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)
        .expect("N128 occupancy");
    println!(
        "symbol=f32_f32_n128_s2 static_shared_bytes={static_shared_bytes} dynamic_shared_bytes=0 local_bytes={local_bytes} registers={registers} occupancy_blocks_per_sm={occupancy}"
    );
    assert_eq!(static_shared_bytes, 49_152);
    assert_eq!(local_bytes, 0);
    assert!(
        registers <= 128,
        "N128 uses {registers} registers per thread"
    );
    assert!(occupancy >= 2, "N128 occupancy is {occupancy} blocks/SM");
}

fn fixed_f32_n128_baseline_window_us(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    iterations: usize,
) -> f64 {
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFma);
    fixed_tile_window_us(ctx, operands, shape, InferenceTile::Legacy, iterations)
}

fn fixed_f32_n128_candidate_window_us(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    iterations: usize,
) -> f64 {
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFma);
    fixed_tile_window_us(ctx, operands, shape, InferenceTile::F32N128S2, iterations)
}

fn fixed_f32_n128_cublas_window_us(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    iterations: usize,
) -> f64 {
    ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record cuBLAS diagnostic start");
    for _ in 0..iterations {
        gpu_gemm_typed_forward_raw(
            ctx,
            operands.c,
            operands.x,
            operands.w,
            None,
            (shape.m, shape.k, shape.n),
        )
        .expect("fast cuBLAS diagnostic launch");
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("record cuBLAS diagnostic end");
    f64::from(start.elapsed_ms(&end).expect("measure cuBLAS diagnostic")) * 1000.0
        / iterations as f64
}

fn f32_n128_calibrated_iterations(pilot_us: f64) -> usize {
    assert!(pilot_us.is_finite() && pilot_us > 0.0);
    (5000.0 / pilot_us).ceil().clamp(1.0, 4096.0) as usize
}

#[test]
#[ignore = "requires a quiet 170-SM CC12 CUDA device and emits paired F32 N128 evidence"]
fn fixed_f32_n128_paired_a_e() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(
        device.compute_capability,
        (12, 0),
        "paired gate requires CC12.0"
    );
    assert_eq!(
        device.multiprocessor_count(),
        170,
        "paired gate requires 170 SMs"
    );
    let ctx = GpuCtx::new(&device).expect("GPU context");
    assert!(ctx.tf32(), "fast cuBLAS diagnostic requires TF32 enabled");
    assert_fixed_f32_n128_resources(&ctx);
    let cells = [
        (
            "A",
            InferenceShape {
                m: 4621,
                k: 384,
                n: 1928,
            },
        ),
        (
            "B",
            InferenceShape {
                m: 4621,
                k: 768,
                n: 2304,
            },
        ),
        (
            "C",
            InferenceShape {
                m: 4621,
                k: 1928,
                n: 384,
            },
        ),
        (
            "D",
            InferenceShape {
                m: 2048,
                k: 768,
                n: 2304,
            },
        ),
        (
            "E",
            InferenceShape {
                m: 2048,
                k: 2304,
                n: 768,
            },
        ),
    ];
    let mut eligible_points = Vec::new();
    let mut rejected_points = Vec::new();
    let mut paired_records = 0;
    let mut cublas_records = 0;
    for (label, shape) in cells {
        let a = f32_n128_offset_buffer(
            &ctx,
            &synth(shape.m * shape.k, 0xa128_ae00 ^ shape.m as u64),
            0,
            "paired A",
        );
        let b = f32_n128_offset_buffer(
            &ctx,
            &synth(shape.k * shape.n, 0xb128_ae00 ^ shape.n as u64),
            0,
            "paired B",
        );
        let baseline = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("paired baseline output");
        let candidate = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("paired candidate output");
        let cublas = DtypedBuf::zeros(&ctx.stream, shape.m * shape.n, WeightDtype::F32)
            .expect("paired cuBLAS output");
        let baseline_ops = InferenceFwdOperands {
            c: typed(&baseline, WeightDtype::F32),
            x: typed(&a, WeightDtype::F32),
            w: typed(&b, WeightDtype::F32),
            bias_ptr: None,
        };
        let candidate_ops = InferenceFwdOperands {
            c: typed(&candidate, WeightDtype::F32),
            ..baseline_ops
        };
        let cublas_ops = InferenceFwdOperands {
            c: typed(&cublas, WeightDtype::F32),
            ..baseline_ops
        };

        inference_forward_f32_legacy_baseline(&ctx, baseline_ops, shape)
            .expect("paired bit oracle launch");
        inference_forward_with_tile(&ctx, candidate_ops, shape, InferenceTile::F32N128S2)
            .expect("paired bit candidate launch");
        ctx.stream.synchronize().expect("paired bit-gate sync");
        assert_eq!(
            f32_bits(&ctx, &candidate, shape.m * shape.n),
            f32_bits(&ctx, &baseline, shape.m * shape.n),
            "paired candidate/oracle bit gate for cell {label}"
        );

        for _ in 0..128 {
            inference_forward_with_tile(&ctx, candidate_ops, shape, InferenceTile::F32N128S2)
                .expect("warm N128 candidate");
        }
        ctx.stream.synchronize().expect("N128 warmup sync");
        for _ in 0..128 {
            inference_forward_with_tile(&ctx, baseline_ops, shape, InferenceTile::Legacy)
                .expect("warm incumbent S2");
        }
        ctx.stream.synchronize().expect("S2 warmup sync");
        let candidate_iterations = f32_n128_calibrated_iterations(
            fixed_f32_n128_candidate_window_us(&ctx, candidate_ops, shape, 16),
        );
        let baseline_iterations = f32_n128_calibrated_iterations(
            fixed_f32_n128_baseline_window_us(&ctx, baseline_ops, shape, 16),
        );

        let mut cell_eligible = true;
        for candidate_first in [true, false] {
            let mut ratios = Vec::with_capacity(101);
            let mut candidate_us = Vec::with_capacity(101);
            let mut baseline_us = Vec::with_capacity(101);
            for _ in 0..101 {
                let (candidate_elapsed, baseline_elapsed) = if candidate_first {
                    (
                        fixed_f32_n128_candidate_window_us(
                            &ctx,
                            candidate_ops,
                            shape,
                            candidate_iterations,
                        ),
                        fixed_f32_n128_baseline_window_us(
                            &ctx,
                            baseline_ops,
                            shape,
                            baseline_iterations,
                        ),
                    )
                } else {
                    let baseline_elapsed = fixed_f32_n128_baseline_window_us(
                        &ctx,
                        baseline_ops,
                        shape,
                        baseline_iterations,
                    );
                    let candidate_elapsed = fixed_f32_n128_candidate_window_us(
                        &ctx,
                        candidate_ops,
                        shape,
                        candidate_iterations,
                    );
                    (candidate_elapsed, baseline_elapsed)
                };
                assert!(
                    candidate_elapsed.is_finite()
                        && candidate_elapsed > 0.0
                        && baseline_elapsed.is_finite()
                        && baseline_elapsed > 0.0,
                    "paired exact-arm latencies must be finite and positive"
                );
                candidate_us.push(candidate_elapsed);
                baseline_us.push(baseline_elapsed);
                ratios.push(candidate_elapsed / baseline_elapsed);
            }
            candidate_us.sort_by(f64::total_cmp);
            baseline_us.sort_by(f64::total_cmp);
            ratios.sort_by(f64::total_cmp);
            assert_eq!(ratios.len(), 101, "paired exact-arm protocol window count");
            let order = if candidate_first {
                "candidate_then_baseline"
            } else {
                "baseline_then_candidate"
            };
            let ratio_p95 = percentile(&ratios, 0.95);
            println!(
                concat!(
                    "{{\"schema\":\"MambaBiFixedF32N128PairedV1\",",
                    "\"cell\":\"{}\",\"m\":{},\"k\":{},\"n\":{},",
                    "\"order\":\"{}\",\"warmups\":128,\"pilot_iterations\":16,",
                    "\"target_window_ms\":5.0,\"pairs\":101,",
                    "\"candidate_iterations\":{},",
                    "\"baseline_iterations\":{},",
                    "\"candidate_p50_us\":{:.9},\"baseline_p50_us\":{:.9},",
                    "\"ratio_p50\":{:.9},\"ratio_p95\":{:.9}}}"
                ),
                label,
                shape.m,
                shape.k,
                shape.n,
                order,
                candidate_iterations,
                baseline_iterations,
                percentile(&candidate_us, 0.50),
                percentile(&baseline_us, 0.50),
                percentile(&ratios, 0.50),
                ratio_p95,
            );
            cell_eligible &= ratio_p95 <= 0.98;
            paired_records += 1;
        }
        if cell_eligible {
            eligible_points.push(label);
        } else {
            rejected_points.push(label);
        }

        ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
        for _ in 0..128 {
            gpu_gemm_typed_forward_raw(
                &ctx,
                cublas_ops.c,
                cublas_ops.x,
                cublas_ops.w,
                None,
                (shape.m, shape.k, shape.n),
            )
            .expect("warm fast cuBLAS diagnostic");
        }
        ctx.stream
            .synchronize()
            .expect("cuBLAS diagnostic warmup sync");
        let cublas_iterations = f32_n128_calibrated_iterations(fixed_f32_n128_cublas_window_us(
            &ctx, cublas_ops, shape, 16,
        ));
        for candidate_first in [true, false] {
            let mut ratios = Vec::with_capacity(101);
            for _ in 0..101 {
                let (candidate_elapsed, cublas_elapsed) = if candidate_first {
                    (
                        fixed_f32_n128_candidate_window_us(
                            &ctx,
                            candidate_ops,
                            shape,
                            candidate_iterations,
                        ),
                        fixed_f32_n128_cublas_window_us(&ctx, cublas_ops, shape, cublas_iterations),
                    )
                } else {
                    let cublas_elapsed =
                        fixed_f32_n128_cublas_window_us(&ctx, cublas_ops, shape, cublas_iterations);
                    let candidate_elapsed = fixed_f32_n128_candidate_window_us(
                        &ctx,
                        candidate_ops,
                        shape,
                        candidate_iterations,
                    );
                    (candidate_elapsed, cublas_elapsed)
                };
                assert!(
                    candidate_elapsed.is_finite()
                        && candidate_elapsed > 0.0
                        && cublas_elapsed.is_finite()
                        && cublas_elapsed > 0.0,
                    "candidate/cuBLAS latencies must be finite and positive"
                );
                ratios.push(candidate_elapsed / cublas_elapsed);
            }
            ratios.sort_by(f64::total_cmp);
            assert_eq!(ratios.len(), 101, "candidate/cuBLAS protocol window count");
            let order = if candidate_first {
                "candidate_then_cublas"
            } else {
                "cublas_then_candidate"
            };
            println!(
                "{{\"schema\":\"MambaBiFixedF32N128CublasDiagnosticV1\",\"cell\":\"{label}\",\"m\":{},\"k\":{},\"n\":{},\"order\":\"{order}\",\"pairs\":101,\"candidate_iterations\":{candidate_iterations},\"cublas_iterations\":{cublas_iterations},\"ratio_p50\":{:.9},\"ratio_p95\":{:.9}}}",
                shape.m,
                shape.k,
                shape.n,
                percentile(&ratios, 0.50),
                percentile(&ratios, 0.95),
            );
            cublas_records += 1;
        }
    }
    assert_eq!(paired_records, 10, "five cells times two exact-arm orders");
    assert_eq!(
        cublas_records, 10,
        "five cells times two candidate/cuBLAS orders"
    );
    assert_eq!(
        eligible_points,
        vec!["A", "B"],
        "only both-order qualifying points may enter AUTO"
    );
    assert_eq!(
        rejected_points,
        vec!["C", "D", "E"],
        "the rejected general-route controls must remain ineligible"
    );
    let decision = r#"{"schema":"MambaBiFixedF32N128Decision","general_route_eligible":false,"eligible_points":[{"cell":"A","m":4621,"k":384,"n":1928,"device_cc":"12.0","sm_count":170},{"cell":"B","m":4621,"k":768,"n":2304,"device_cc":"12.0","sm_count":170}],"rejected_points":["C","D","E"],"required_p95_max":0.98}"#;
    println!("{decision}");
}
