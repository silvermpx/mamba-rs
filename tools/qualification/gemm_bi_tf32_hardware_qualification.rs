//! The TF32 hardware gate: build the qualification binary, run the full
//! runtime and performance corpus on a board of the exact compute
//! capability, verify the report's digests, driver ABI proof and symbol
//! inventory, then rerun it under every compute-sanitizer tool. One arm per
//! architecture; each requires its own board, so every arm is ignored and
//! run by name with `--features cuda,qualification`.

#![cfg(feature = "cuda")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

#[path = "../../tests/support/gemm_bi_tf32_contract_toolkit.rs"]
mod toolkit;
use mamba_rs::mamba_ssm::gpu::GemmMode;
use toolkit::*;

fn require_exact_cc(expected: (u32, u32)) {
    let device = mamba_rs::mamba_ssm::gpu::device::GpuDevice::new(0)
        .expect("open the exact qualification GPU");
    assert_eq!(
        device.compute_capability, expected,
        "wrong qualification CC"
    );
}
fn json_string_field<'a>(fields: &'a BTreeMap<String, StrictJsonValue>, field: &str) -> &'a str {
    strict_json_string(fields, field)
        .unwrap_or_else(|error| panic!("{error}"))
        .0
}
fn assert_hex_digest(fields: &BTreeMap<String, StrictJsonValue>, field: &str) {
    let digest = json_string_field(fields, field);
    assert_eq!(digest.len(), 64, "{field} must contain 256 bits");
    assert!(
        digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} must be canonical lowercase hexadecimal: {digest}"
    );
    assert!(
        digest.bytes().any(|byte| byte != b'0'),
        "{field} may not use the all-zero sentinel"
    );
}
fn assert_driver_abi_proof(proof: &str, expected: &BTreeSet<String>, tensor_map_alignment: usize) {
    let mut lines = proof.lines();
    assert_eq!(
        lines.next(),
        Some("MambaBiTf32DriverAbiV2"),
        "driver ABI proof schema"
    );
    let mut records = BTreeMap::new();
    for line in lines {
        let mut fields = line.split('\t');
        let symbol = fields.next().expect("driver ABI symbol");
        let count: usize = fields
            .next()
            .expect("driver ABI parameter count")
            .parse()
            .expect("numeric driver ABI parameter count");
        let count_source = fields.next().expect("driver ABI count source");
        let layout = fields.next().expect("driver ABI parameter layout");
        assert!(fields.next().is_none(), "extra driver ABI fields: {line}");
        // Stream-K carries the partial slabs and the flags ahead of the maps.
        let streamk = symbol.contains("_streamk");
        assert_eq!(
            count,
            if streamk { 7 } else { 5 },
            "{symbol} live Driver ABI parameter count"
        );
        assert_eq!(
            count_source, "ptx_contract+cuFuncGetParamInfo_terminal_probe",
            "{symbol} live Driver ABI count source"
        );
        let expected_layout = if symbol.contains("_sm80_") {
            "0:8,8:8,16:8,24:8,32:32"
        } else if streamk && tensor_map_alignment == 64 {
            "0:8,8:8,16:8,64:128,192:128,320:8,328:40"
        } else if streamk {
            "0:8,8:8,16:8,128:128,256:128,384:8,392:40"
        } else if tensor_map_alignment == 64 {
            "0:8,64:128,192:128,320:8,328:40"
        } else {
            "0:8,128:128,256:128,384:8,392:40"
        };
        assert_eq!(layout, expected_layout, "{symbol} cuFuncGetParamInfo");
        assert!(
            records.insert(symbol.to_owned(), ()).is_none(),
            "duplicate driver ABI proof for {symbol}"
        );
    }
    assert_eq!(
        records.keys().cloned().collect::<BTreeSet<_>>(),
        *expected,
        "driver ABI proof inventory"
    );
}
fn run_hardware_qualification(cc: (u32, u32), expected_routes: usize) {
    require_exact_cc(cc);
    let census = qualified_op_census(cc, expected_routes);
    let directory = tempfile::tempdir().expect("TF32 qualification target directory");
    let target = directory.path().join("target");
    let mut build = Command::new("cargo");
    build.args([
        "build",
        "--features",
        "cuda,qualification",
        "--bin",
        "gemm-bi-tf32-qualification",
        "--target-dir",
    ]);
    build.arg(&target);
    checked_output(build, "build gemm-bi-tf32-qualification");

    let binary = target.join("debug/gemm-bi-tf32-qualification");
    assert!(binary.is_file(), "missing qualification binary {binary:?}");
    let artifact_output = directory.path().join("qualification-artifact.bin");
    let driver_abi_output = directory.path().join("driver-abi.tsv");
    let arguments =
        qualification_arguments(cc, expected_routes, &artifact_output, &driver_abi_output);
    let mut qualification = Command::new(&binary);
    qualification.args(&arguments);
    let output = checked_output(qualification, "TF32 runtime qualification");
    let report = String::from_utf8(output.stdout).expect("qualification JSON must be UTF-8");
    assert_eq!(
        report.trim(),
        report,
        "qualification report may not contain framing whitespace"
    );
    assert!(
        report.starts_with('{') && report.ends_with('}') && !report.contains('\n'),
        "qualification report must be exactly one JSON object"
    );
    let parsed = parse_strict_json(&report)
        .unwrap_or_else(|error| panic!("invalid qualification JSON: {error}: {report}"));
    let fields = strict_json_object(&parsed).expect("qualification JSON object");
    assert_eq!(
        json_string_field(fields, "schema"),
        "MambaBiTf32QualificationV5"
    );
    assert_eq!(
        json_string_field(fields, "exact_cc"),
        format!("{}.{}", cc.0, cc.1)
    );
    assert_eq!(json_string_field(fields, "suite"), "full");
    for field in [
        "runtime",
        "primary_accuracy",
        "boundary_accuracy",
        "repeat_determinism",
        "graph_replay",
        "eager_graph_route_identity",
        "driver_abi",
        "k0_runtime",
        "k0_epilogue_bits",
        "staged_k_reuse",
        "strided_subviews",
        "red_zone_canaries",
        "epilogue_scalars",
        "exceptional_values",
        "cross_m_batch_invariance",
        "driver_jit_local_memory",
    ] {
        assert_eq!(
            json_string_field(fields, field),
            "pass",
            "qualification gate {field} did not pass"
        );
    }
    assert_eq!(json_string_field(fields, "timings"), "recorded");
    let (expected_max_local_bytes, expected_local_exceptions) = (0, 0);
    for (field, expected) in [
        ("repeat", 100_u64),
        (
            "max_driver_jit_local_memory_bytes",
            expected_max_local_bytes,
        ),
        (
            "driver_jit_local_memory_exception_count",
            expected_local_exceptions,
        ),
        ("routes_qualified", expected_routes as u64),
        ("k0_symbols_qualified", expected_routes as u64),
        ("driver_abi_entries", expected_routes as u64),
        ("boundary_cases", (expected_routes * 47) as u64),
        ("staged_cases", (expected_routes * 4) as u64),
        ("guard_poison_pairs", expected_routes as u64),
        ("exceptional_symbols", expected_routes as u64),
        ("cross_m_symbols", (census.nn + census.nt) as u64),
        ("cross_m_shapes", ((census.nn + census.nt) * 4) as u64),
        ("nn_bias_beta_symbols", census.nn as u64),
    ] {
        assert_eq!(
            strict_json_u64(fields, field).unwrap_or_else(|error| panic!("{error}")),
            expected,
            "qualification count {field}"
        );
    }
    let digest_fields = [
        "artifact_digest",
        "output_digest",
        "zero_reduction_digest",
        "boundary_output_digest",
        "tensor_map_identity_digest",
        "ordered_graph_route_digest",
        "staged_guarded_digest",
        "exceptional_values_digest",
        "cross_m_invariance_digest",
        "driver_abi_digest",
        "driver_jit_resource_digest",
        "report_digest",
    ];
    for field in digest_fields {
        assert_hex_digest(fields, field);
    }
    for (index, field) in digest_fields.iter().enumerate() {
        for other in &digest_fields[index + 1..] {
            assert_ne!(
                json_string_field(fields, field),
                json_string_field(fields, other),
                "{field} and {other} must use distinct digest domains"
            );
        }
    }
    let artifact = std::fs::read(&artifact_output)
        .unwrap_or_else(|error| panic!("read qualification artifact {artifact_output:?}: {error}"));
    let expected_symbols = expected_hardware_symbols(cc);
    assert_eq!(expected_symbols.len(), expected_routes);
    let driver_abi_proof = std::fs::read_to_string(&driver_abi_output)
        .unwrap_or_else(|error| panic!("read driver ABI proof {driver_abi_output:?}: {error}"));
    verify_qualification_digests(&report, &artifact, driver_abi_proof.as_bytes())
        .unwrap_or_else(|error| panic!("qualification digest verification failed: {error}"));
    let tensor_map_alignment = match loaded_nvrtc_version().0 {
        12 => 64,
        13 => 128,
        version => panic!("unsupported CUDA tensor-map ABI major {version}"),
    };
    assert_driver_abi_proof(&driver_abi_proof, &expected_symbols, tensor_map_alignment);
    let runner_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/gemm_bi_tf32_qualification.rs"),
    )
    .expect("read production TF32 qualification runner");
    assert_code_contains_all(
        &runner_source,
        &["run_tf32_qualification"],
        "live Driver API ABI qualification",
    );
    assert_cuda12_driver_abi_lookup_contract();

    for tool in ["memcheck", "racecheck", "initcheck", "synccheck"] {
        let sanitizer_arguments =
            sanitizer_qualification_arguments(cc, expected_routes, directory.path(), tool);
        let sanitizer = sanitizer_command(&binary, &sanitizer_arguments, tool);
        checked_output(sanitizer, &format!("compute-sanitizer {tool}"));
    }
}
fn nt_fixed_split2_sample_bits(
    a: &[u32],
    b: &[u32],
    row: usize,
    column: usize,
    reduction: usize,
) -> u32 {
    let split_boundary = reduction.div_ceil(16).div_ceil(2) * 16;
    let mut partials = [0.0_f32; 2];
    for (split, range) in [
        0..split_boundary.min(reduction),
        split_boundary.min(reduction)..reduction,
    ]
    .into_iter()
    .enumerate()
    {
        for index in range {
            partials[split] = f32::from_bits(a[row * reduction + index]).mul_add(
                f32::from_bits(b[column * reduction + index]),
                partials[split],
            );
        }
    }
    (0.0_f32 + partials[0] + partials[1]).to_bits()
}
#[test]
#[ignore = "requires exact CC 12.0/170SM with CUDA 13.0 measured-symbol exclusion"]
fn sm120_exact_nt_d768_out_reachable_fallback_matches_fixed_split_cpu_bits() -> Result<(), String> {
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationF32Epilogue, PhysicalQualificationRequest, PhysicalQualificationRoute,
        presize_physical_qualification_suite, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ResolvedGemmOp, ResolvedNumericContract, ResolvedOutputOwnership,
    };

    const DIMS: (usize, usize, usize) = (2_048, 1_536, 768);
    const SALT: u64 = 0x1205_1300;
    const SYMBOL: &str = "gemm_bi_nt_sm120_tma_fma_v1_m128n64_bk16_s2_kvec";
    let device = GpuDevice::new(0)?;
    if device.compute_capability != (12, 0) || device.multiprocessor_count() != 170 {
        return Err(format!(
            "fallback probe requires exact CC 12.0/170SM, got {:?}/{}SM",
            device.compute_capability,
            device.multiprocessor_count()
        ));
    }
    let ctx = GpuCtx::new(&device)?;
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
    let specialized = ctx
        .kernels
        .f32_triad_availability()
        .specialized
        .ok_or_else(|| "fallback probe has no specialized TF32 binding".to_string())?;
    if specialized.compiler.nvrtc_version != (13, 0) {
        return Err(format!(
            "fallback probe requires CUDA 13.0 NVRTC, got {:?}",
            specialized.compiler.nvrtc_version
        ));
    }
    let request = PhysicalQualificationRequest::contiguous_f32(
        ResolvedGemmOp::Nt,
        DIMS,
        PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
        PhysicalQualificationF32Epilogue::new(1.0, 0.0, false),
    );
    presize_physical_qualification_suite(&ctx, &[request])?;
    let mut launch = qualify_physical_launch(&ctx, request)?;
    let nodes = launch.evidence().nodes();
    if nodes.len() != 1 {
        return Err(format!("fallback probe expected one node, got {nodes:?}"));
    }
    let node = nodes[0];
    if node.symbol != SYMBOL
        || node.tile != Some((128, 64))
        || node.launch.grid_dim != (768, 1, 1)
        || node.numeric_contract != Some(ResolvedNumericContract::ScalarFmaFixedSplitFoldV1)
        || node.ownership != Some(ResolvedOutputOwnership::OwnerCtaPerOutputTileFixedSplitFoldV1)
    {
        return Err(format!(
            "fallback probe selected the wrong same-split route: {node:?}"
        ));
    }

    launch.seed_f32_operands(&ctx, SALT)?;
    let (a, b) = launch.f32_operand_bits(&ctx)?;
    launch.measure_eager_window_ms(&ctx, 1)?;
    let eager = launch.f32_output_bits(&ctx)?;
    for (row, column) in [
        (0, 0),
        (0, 63),
        (0, 64),
        (127, 127),
        (128, 128),
        (511, 383),
        (1_024, 768),
        (2_047, 1_535),
    ] {
        let actual = eager[row * DIMS.1 + column];
        let expected = nt_fixed_split2_sample_bits(&a, &b, row, column, DIMS.2);
        if actual != expected {
            return Err(format!(
                "fallback fixed-split bits differ at ({row},{column}): actual={actual:#010x} expected={expected:#010x}"
            ));
        }
    }

    launch.seed_f32_operands(&ctx, SALT)?;
    launch.measure_eager_window_ms(&ctx, 1)?;
    if launch.f32_output_bits(&ctx)? != eager {
        return Err("fallback eager repeat changed output bits".into());
    }
    launch.seed_f32_operands(&ctx, SALT)?;
    launch.measure_graph_window_ms(&ctx, 1)?;
    if launch.f32_output_bits(&ctx)? != eager {
        return Err("fallback eager and graph output bits differ".into());
    }
    launch.validate_red_zones(&ctx)?;
    Ok(())
}
#[test]
#[ignore = "requires exact CC 8.0 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm80_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((8, 0), 19);
}
#[test]
#[ignore = "requires exact CC 8.6 and the full portable TF32 qualification corpus"]
fn hardware_sm86_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((8, 6), 19);
}
#[test]
#[ignore = "requires exact CC 8.7 and the full portable TF32 qualification corpus"]
fn hardware_sm87_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((8, 7), 19);
}
#[test]
#[ignore = "requires exact Ada CC 8.9 and the full portable TF32 qualification corpus"]
fn hardware_sm89_ada_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((8, 9), 19);
}
#[test]
#[ignore = "requires exact CC 9.0 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm90a_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((9, 0), 25);
}
#[test]
#[ignore = "requires exact CC 10.0 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm100_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((10, 0), 55);
}
#[test]
#[ignore = "requires exact CC 10.3 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm103_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((10, 3), 55);
}
#[test]
#[ignore = "requires exact CC 11.0 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm110_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((11, 0), 55);
}
#[test]
#[ignore = "requires exact CC 12.0 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm120_tf32_runtime_and_performance_gate() {
    // Eighteen portable routes plus the eighteen SM120 TF32 routes; the
    // exact-F32 SM120 routes are not TF32 qualification candidates.
    run_hardware_qualification((12, 0), 36);
}
#[test]
#[ignore = "requires exact CC 12.1 and the full TF32 runtime/performance qualification corpus"]
fn hardware_sm121_tf32_runtime_and_performance_gate() {
    run_hardware_qualification((12, 1), 36);
}
