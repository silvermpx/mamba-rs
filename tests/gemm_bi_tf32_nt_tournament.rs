#[cfg(feature = "cuda")]
mod common;

const DISCOVERY_WINDOWS: usize = 21;
const FINAL_WINDOWS: usize = 101;
const SCHEMA: &str = "MambaBiTf32NtTournamentV1";
const DECISION: &str = "measurement_only_no_dispatcher_admission";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    name: &'static str,
    dims: (usize, usize, usize),
}

const CELLS: [Cell; 2] = [
    Cell {
        name: "live_m64_k384_n1536",
        dims: (64, 384, 1_536),
    },
    Cell {
        name: "reverse_m384_k64_n1536",
        dims: (384, 64, 1_536),
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ArmSpec {
    name: &'static str,
    symbol: &'static str,
    tile: (u32, u32),
    partitions: u32,
    stages: u8,
    dynamic_shared_bytes: u32,
    register_cap: u32,
    occupancy_gate: u32,
}

const ARMS: [ArmSpec; 4] = [
    ArmSpec {
        name: "p4_s3",
        symbol: "gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s3",
        tile: (16, 32),
        partitions: 4,
        stages: 3,
        dynamic_shared_bytes: 20_736,
        register_cap: 96,
        occupancy_gate: 3,
    },
    ArmSpec {
        name: "p4_s4",
        symbol: "gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4",
        tile: (16, 32),
        partitions: 4,
        stages: 4,
        dynamic_shared_bytes: 27_648,
        register_cap: 96,
        occupancy_gate: 3,
    },
    ArmSpec {
        name: "p8_s3",
        symbol: "gemm_bi_nt_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s3",
        tile: (32, 32),
        partitions: 8,
        stages: 3,
        dynamic_shared_bytes: 27_648,
        register_cap: 96,
        occupancy_gate: 3,
    },
    ArmSpec {
        name: "p8_s4",
        symbol: "gemm_bi_nt_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s4",
        tile: (32, 32),
        partitions: 8,
        stages: 4,
        dynamic_shared_bytes: 36_864,
        register_cap: 96,
        occupancy_gate: 2,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExpectedManifest {
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
}

fn tournament_cells() -> &'static [Cell] {
    &CELLS
}

fn tournament_arms() -> &'static [ArmSpec] {
    &ARMS
}

fn window_count(value: Option<&str>) -> Result<usize, String> {
    match value {
        None | Some("discovery") | Some("21") => Ok(DISCOVERY_WINDOWS),
        Some("final") | Some("101") => Ok(FINAL_WINDOWS),
        Some(value) => Err(format!(
            "MAMBA_RS_TF32_NT_TOURNAMENT_FINAL must be discovery/21 or final/101, received {value:?}"
        )),
    }
}

fn latin_order(window: usize, reverse: bool) -> Vec<usize> {
    let mut order = (0..ARMS.len()).collect::<Vec<_>>();
    order.rotate_left(window % ARMS.len());
    if reverse {
        order.reverse();
    }
    order
}

fn expected_manifest(cell: Cell, arm: ArmSpec) -> ExpectedManifest {
    ExpectedManifest {
        grid_dim: (
            (cell.dims.1 as u32).div_ceil(arm.tile.1),
            (cell.dims.0 as u32).div_ceil(arm.tile.0),
            arm.partitions,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: arm.dynamic_shared_bytes,
    }
}

struct CompletionRecord<'a> {
    record_type: &'static str,
    records_before_completion: usize,
    records_total: usize,
    windows_per_order: usize,
    content_sha256_before_completion: &'a str,
    dataset_complete: bool,
    measurement_only: bool,
    runtime_autotuning: bool,
    dispatcher_admitted: bool,
    decision: &'static str,
}

impl CompletionRecord<'_> {
    fn render(&self) -> String {
        format!(
            concat!(
                "{{\"schema\":\"{}\",\"record_type\":\"{}\",",
                "\"records_before_completion\":{},\"records_total\":{},",
                "\"windows_per_order\":{},\"content_sha256_before_completion\":\"{}\",",
                "\"dataset_complete\":{},\"measurement_only\":{},",
                "\"runtime_autotuning\":{},\"dispatcher_admitted\":{},",
                "\"decision\":\"{}\"}}"
            ),
            SCHEMA,
            self.record_type,
            self.records_before_completion,
            self.records_total,
            self.windows_per_order,
            self.content_sha256_before_completion,
            self.dataset_complete,
            self.measurement_only,
            self.runtime_autotuning,
            self.dispatcher_admitted,
            self.decision,
        )
    }
}

fn completion_record(
    records_before_completion: usize,
    windows_per_order: usize,
    content_sha256_before_completion: &str,
) -> CompletionRecord<'_> {
    CompletionRecord {
        record_type: "completion",
        records_before_completion,
        records_total: records_before_completion + 1,
        windows_per_order,
        content_sha256_before_completion,
        dataset_complete: true,
        measurement_only: true,
        runtime_autotuning: false,
        dispatcher_admitted: false,
        decision: DECISION,
    }
}

#[test]
fn tournament_matrix_freezes_two_shapes_and_four_forced_routes() {
    assert_eq!(
        tournament_cells()
            .iter()
            .map(|cell| (cell.name, cell.dims))
            .collect::<Vec<_>>(),
        vec![
            ("live_m64_k384_n1536", (64, 384, 1_536)),
            ("reverse_m384_k64_n1536", (384, 64, 1_536)),
        ]
    );
    assert_eq!(
        tournament_arms()
            .iter()
            .map(|arm| {
                (
                    arm.name,
                    arm.symbol,
                    arm.tile,
                    arm.partitions,
                    arm.stages,
                    arm.dynamic_shared_bytes,
                    arm.register_cap,
                    arm.occupancy_gate,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (
                "p4_s3",
                "gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s3",
                (16, 32),
                4,
                3,
                20_736,
                96,
                3,
            ),
            (
                "p4_s4",
                "gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4",
                (16, 32),
                4,
                4,
                27_648,
                96,
                3,
            ),
            (
                "p8_s3",
                "gemm_bi_nt_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s3",
                (32, 32),
                8,
                3,
                27_648,
                96,
                3,
            ),
            (
                "p8_s4",
                "gemm_bi_nt_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s4",
                (32, 32),
                8,
                4,
                36_864,
                96,
                2,
            ),
        ]
    );
}

#[test]
fn discovery_and_final_window_modes_are_explicit_and_fail_closed() {
    assert_eq!(window_count(None).unwrap(), 21);
    assert_eq!(window_count(Some("discovery")).unwrap(), 21);
    assert_eq!(window_count(Some("21")).unwrap(), 21);
    assert_eq!(window_count(Some("final")).unwrap(), 101);
    assert_eq!(window_count(Some("101")).unwrap(), 101);
    for invalid in ["", "1", "true", "20", "100", "102"] {
        assert!(window_count(Some(invalid)).is_err(), "accepted {invalid:?}");
    }
}

#[test]
fn latin_reverse_schedule_balances_every_arm_and_position() {
    let mut forward_positions = [[0usize; 4]; 4];
    let mut reverse_positions = [[0usize; 4]; 4];
    for window in 0..4 {
        let forward = latin_order(window, false);
        let reverse = latin_order(window, true);
        assert_eq!(reverse, forward.iter().rev().copied().collect::<Vec<_>>());
        for (position, arm) in forward.into_iter().enumerate() {
            forward_positions[arm][position] += 1;
        }
        for (position, arm) in reverse.into_iter().enumerate() {
            reverse_positions[arm][position] += 1;
        }
    }
    assert!(
        forward_positions
            .into_iter()
            .flatten()
            .all(|count| count == 1)
    );
    assert!(
        reverse_positions
            .into_iter()
            .flatten()
            .all(|count| count == 1)
    );
}

#[test]
fn expected_manifests_cover_every_cell_and_arm() {
    let manifests = tournament_cells()
        .iter()
        .flat_map(|cell| {
            tournament_arms()
                .iter()
                .map(move |arm| (cell.name, arm.name, expected_manifest(*cell, *arm)))
        })
        .collect::<Vec<_>>();
    assert_eq!(manifests.len(), 8);
    assert_eq!(manifests[0].2.grid_dim, (12, 4, 4));
    assert_eq!(manifests[2].2.grid_dim, (12, 2, 8));
    assert_eq!(manifests[4].2.grid_dim, (2, 24, 4));
    assert_eq!(manifests[6].2.grid_dim, (2, 12, 8));
    assert!(
        manifests
            .iter()
            .all(|entry| entry.2.block_dim == (128, 1, 1))
    );
}

#[test]
fn completion_record_is_measurement_only_and_never_dispatcher_admission() {
    let record = completion_record(4, 21, "abc123");
    assert_eq!(record.record_type, "completion");
    assert_eq!(record.records_before_completion, 4);
    assert_eq!(record.records_total, 5);
    assert_eq!(record.windows_per_order, 21);
    assert_eq!(record.content_sha256_before_completion, "abc123");
    assert!(record.dataset_complete);
    assert!(record.measurement_only);
    assert!(!record.runtime_autotuning);
    assert!(!record.dispatcher_admitted);
    assert_eq!(record.decision, "measurement_only_no_dispatcher_admission");
    assert_eq!(
        record.render(),
        concat!(
            "{\"schema\":\"MambaBiTf32NtTournamentV1\",\"record_type\":\"completion\",",
            "\"records_before_completion\":4,\"records_total\":5,\"windows_per_order\":21,",
            "\"content_sha256_before_completion\":\"abc123\",\"dataset_complete\":true,",
            "\"measurement_only\":true,\"runtime_autotuning\":false,",
            "\"dispatcher_admitted\":false,",
            "\"decision\":\"measurement_only_no_dispatcher_admission\"}"
        )
    );
}

#[cfg(feature = "cuda")]
mod cuda_suite {
    use std::fmt::Write as _;
    use std::fs::{File, OpenOptions};
    use std::io::{BufWriter, Write as _};
    use std::path::PathBuf;

    use super::*;
    use common::gpu_quiet::QuietGpu;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationRequest, PhysicalQualificationRoute, QualifiedPhysicalLaunch,
        Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile,
        presize_physical_qualification_suite, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ModuleKind, PhysicalLaunchKind, PolicyDtype, ResolvedGemmOp,
    };
    use sha2::{Digest as _, Sha256};

    const OUTPUT_ENV: &str = "MAMBA_RS_TF32_NT_TOURNAMENT_JSONL";
    const FINAL_ENV: &str = "MAMBA_RS_TF32_NT_TOURNAMENT_FINAL";
    const SALT: u64 = 0x6e74_7038;
    const DETERMINISM_REPEATS: usize = 10;
    const WARMUPS: usize = 64;
    const PILOT_ITERATIONS: usize = 16;
    const TARGET_WINDOW_US: f64 = 5_000.0;
    const MAX_WINDOW_ITERATIONS: usize = 4_096;
    const RECORDS_BEFORE_COMPLETION: usize = CELLS.len() * 2;
    const TOURNAMENT_SOURCE: &str = include_str!("gemm_bi_tf32_nt_tournament.rs");
    const CUDA_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm80.cu");

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum PathKind {
        Eager,
        Graph,
    }

    impl PathKind {
        const fn name(self) -> &'static str {
            match self {
                Self::Eager => "eager",
                Self::Graph => "graph",
            }
        }
    }

    impl ArmSpec {
        const fn route(self) -> Tf32PhysicalRoute {
            let tile = match self.tile {
                (16, 32) => Tf32PortableTile::M16N32,
                (32, 32) => Tf32PortableTile::M32N32,
                _ => unreachable!(),
            };
            let stages = match self.stages {
                3 => Tf32PortableStages::S3,
                4 => Tf32PortableStages::S4,
                _ => unreachable!(),
            };
            let portable = Tf32PortableRoute { tile, stages };
            match self.partitions {
                4 => Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(portable),
                8 => Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(portable),
                _ => unreachable!(),
            }
        }
    }

    #[derive(Clone, Copy)]
    struct Percentiles {
        p05: f64,
        p50: f64,
        p95: f64,
    }

    struct OrderedSamples {
        arm_us: [Vec<f64>; 4],
    }

    impl OrderedSamples {
        fn with_capacity(windows: usize) -> Self {
            Self {
                arm_us: std::array::from_fn(|_| Vec::with_capacity(windows)),
            }
        }

        fn push(&mut self, values: [f64; 4]) -> Result<(), String> {
            for (samples, value) in self.arm_us.iter_mut().zip(values) {
                require_positive_finite("timing sample", value)?;
                samples.push(value);
            }
            Ok(())
        }
    }

    struct TournamentSamples {
        forward: OrderedSamples,
        reverse: OrderedSamples,
    }

    struct TournamentSummary {
        forward: [Percentiles; 4],
        reverse: [Percentiles; 4],
        measured_winner: Option<usize>,
    }

    struct CorrectnessEvidence {
        reference_bits: [Vec<u32>; 4],
        output_digests: [[u8; 32]; 4],
        oracle_digest: [u8; 32],
    }

    struct CohortEvidence {
        calibration_preflight: String,
        timed_preflight: String,
        postflight: String,
        iterations: [usize; 4],
    }

    struct RunMetadata {
        pre_context: String,
        gpu_uuid: String,
        compute_capability: (u32, u32),
        multiprocessors: u32,
        driver_api_version: i32,
        driver_build_digest: String,
        nvrtc_version: (i32, i32),
        nvrtc_target: String,
        compile_key: String,
        artifact_digest: String,
        invocation_digest: String,
        tournament_source_sha256: String,
        cuda_source_sha256: String,
        run_identity: String,
    }

    impl RunMetadata {
        fn new(
            launch: &QualifiedPhysicalLaunch<'_>,
            device: &GpuDevice,
            quiet: &QuietGpu,
            pre_context: String,
        ) -> Self {
            let route_identity = launch.evidence().route_identity();
            let compiler = route_identity.compiler;
            let artifact = route_identity.artifacts.triad_sm80;
            let identity = device.identity();
            let tournament_source_sha256 = sha256_bytes(TOURNAMENT_SOURCE.as_bytes());
            let cuda_source_sha256 = sha256_bytes(CUDA_SOURCE.as_bytes());
            let mut run = Sha256::new();
            run.update(b"gemm-bi-tf32-nt-tournament-run.v1");
            run.update(compiler.source_digest);
            run.update(compiler.invocation_digest);
            run.update(artifact.compile_key);
            run.update(artifact.artifact_digest);
            run.update(identity.compute_capability.0.to_le_bytes());
            run.update(identity.compute_capability.1.to_le_bytes());
            run.update(identity.multiprocessor_count.to_le_bytes());
            run.update(identity.driver.api_version.to_le_bytes());
            run.update(identity.driver.build_digest);
            run.update(tournament_source_sha256.as_bytes());
            Self {
                pre_context,
                gpu_uuid: quiet.uuid().to_owned(),
                compute_capability: identity.compute_capability,
                multiprocessors: identity.multiprocessor_count,
                driver_api_version: identity.driver.api_version,
                driver_build_digest: hex_digest(&identity.driver.build_digest),
                nvrtc_version: compiler.nvrtc_version,
                nvrtc_target: compiler.target.as_str().to_owned(),
                compile_key: hex_digest(&artifact.compile_key),
                artifact_digest: hex_digest(&artifact.artifact_digest),
                invocation_digest: hex_digest(&compiler.invocation_digest),
                tournament_source_sha256,
                cuda_source_sha256,
                run_identity: format!("{:x}", run.finalize()),
            }
        }
    }

    struct MeasurementRecord<'a> {
        metadata: &'a RunMetadata,
        cell: Cell,
        path: PathKind,
        windows: usize,
        launches: &'a [QualifiedPhysicalLaunch<'a>],
        correctness: &'a CorrectnessEvidence,
        samples: &'a TournamentSamples,
        summary: &'a TournamentSummary,
        cohort: &'a CohortEvidence,
    }

    struct CohortRun<'ctx, 'run> {
        cell: Cell,
        path: PathKind,
        quiet: &'run QuietGpu,
        contexts: &'ctx [GpuCtx],
        launches: &'run mut [QualifiedPhysicalLaunch<'ctx>],
        correctness: &'run CorrectnessEvidence,
        windows: usize,
    }

    struct JsonlSink {
        path: PathBuf,
        writer: BufWriter<File>,
        digest: Sha256,
        records: usize,
    }

    impl JsonlSink {
        fn create_from_env() -> Result<Self, String> {
            let value = std::env::var_os(OUTPUT_ENV)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| format!("{OUTPUT_ENV} must name a new JSONL file"))?;
            let path = PathBuf::from(value);
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| format!("create TF32 NT evidence {path:?}: {error}"))?;
            if !file
                .metadata()
                .map_err(|error| format!("inspect TF32 NT evidence {path:?}: {error}"))?
                .file_type()
                .is_file()
            {
                return Err(format!("TF32 NT evidence {path:?} is not a regular file"));
            }
            Ok(Self {
                path,
                writer: BufWriter::new(file),
                digest: Sha256::new(),
                records: 0,
            })
        }

        fn write(&mut self, record: String) -> Result<(), String> {
            let mut bytes = record.into_bytes();
            bytes.push(b'\n');
            self.writer
                .write_all(&bytes)
                .map_err(|error| format!("write TF32 NT evidence {:?}: {error}", self.path))?;
            self.digest.update(&bytes);
            self.records += 1;
            Ok(())
        }

        fn finish(mut self, windows: usize) -> Result<(), String> {
            if self.records != RECORDS_BEFORE_COMPLETION {
                return Err(format!(
                    "TF32 NT evidence requires {RECORDS_BEFORE_COMPLETION} measurement records, received {}",
                    self.records
                ));
            }
            let content_digest = format!("{:x}", self.digest.clone().finalize());
            self.write(completion_record(self.records, windows, &content_digest).render())?;
            self.writer
                .flush()
                .map_err(|error| format!("flush TF32 NT evidence {:?}: {error}", self.path))?;
            self.writer
                .get_ref()
                .sync_all()
                .map_err(|error| format!("sync TF32 NT evidence {:?}: {error}", self.path))?;
            eprintln!(
                "TF32 NT tournament wrote {} records to {:?}",
                self.records, self.path
            );
            Ok(())
        }
    }

    fn seeded_values(len: usize, salt: u64) -> Vec<f32> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt;
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let signed = ((state.wrapping_add(index as u64) % 4093) as i32) - 2046;
                signed as f32 / 1024.0
            })
            .collect()
    }

    fn tf32_value(value: f32) -> f64 {
        let bits = value.to_bits();
        let exponent = bits & 0x7f80_0000;
        if exponent == 0x7f80_0000 {
            return value as f64;
        }
        let lsb = (bits >> 13) & 1;
        let rounded = bits.wrapping_add(0x0fff + lsb) & 0xffff_e000;
        f32::from_bits(rounded) as f64
    }

    fn nt_reference(dims: (usize, usize, usize), salt: u64) -> Result<Vec<f64>, String> {
        let (m, k, n) = dims;
        let a_len = m
            .checked_mul(n)
            .ok_or_else(|| "TF32 NT oracle M*N overflows usize".to_string())?;
        let b_len = k
            .checked_mul(n)
            .ok_or_else(|| "TF32 NT oracle K*N overflows usize".to_string())?;
        let output_len = m
            .checked_mul(k)
            .ok_or_else(|| "TF32 NT oracle M*K overflows usize".to_string())?;
        let a = seeded_values(a_len, salt ^ 0x2d);
        let b = seeded_values(b_len, salt ^ 0x67);
        let mut expected = Vec::with_capacity(output_len);
        for row in 0..m {
            for column in 0..k {
                let mut sum = 0.0_f64;
                for inner in 0..n {
                    sum += tf32_value(a[row * n + inner]) * tf32_value(b[column * n + inner]);
                }
                expected.push(sum);
            }
        }
        Ok(expected)
    }

    fn check_numeric(actual: &[f32], expected: &[f64], label: &str) -> Result<(), String> {
        if actual.len() != expected.len() {
            return Err(format!(
                "{label} output length {} differs from CPU reference {}",
                actual.len(),
                expected.len()
            ));
        }
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let actual = f64::from(actual);
            let tolerance = 0.0025 * (1.0 + expected.abs());
            if !actual.is_finite() || (actual - expected).abs() > tolerance {
                return Err(format!(
                    "{label} output {index} was {actual}, CPU reference {expected}, tolerance {tolerance}"
                ));
            }
        }
        Ok(())
    }

    fn context(device: &GpuDevice) -> Result<GpuCtx, String> {
        let ctx = GpuCtx::new(device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_fast_gemm(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
        Ok(ctx)
    }

    fn request(cell: Cell, arm: ArmSpec) -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nt,
            cell.dims,
            PhysicalQualificationRoute::Tf32Forced(arm.route()),
        )
    }

    fn validate_release_build() -> Result<(), String> {
        if cfg!(debug_assertions) {
            Err("TF32 NT tournament requires --release".into())
        } else {
            Ok(())
        }
    }

    fn validate_device(device: &GpuDevice) -> Result<(), String> {
        if device.compute_capability.0 >= 8 {
            Ok(())
        } else {
            Err(format!(
                "TF32 NT tournament requires SM80+, received SM{}.{}",
                device.compute_capability.0, device.compute_capability.1
            ))
        }
    }

    fn assert_manifest(
        cell: Cell,
        arm: ArmSpec,
        launch: &QualifiedPhysicalLaunch<'_>,
    ) -> Result<(), String> {
        let evidence = launch.evidence();
        let expected = expected_manifest(cell, arm);
        let [node] = evidence.nodes() else {
            return Err(format!(
                "{} {} requires exactly one physical node, observed {:?}",
                cell.name,
                arm.name,
                evidence.nodes()
            ));
        };
        let expected_strides = (cell.dims.2, cell.dims.2, cell.dims.1);
        if !evidence.eager_graph_equal()
            || evidence.launch_count() != 1
            || evidence.single_launch_symbol() != Some(arm.symbol)
            || evidence.single_launch_tile() != Some(arm.tile)
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadSm80)
            || evidence.uniform_execution_dtype() != Some(PolicyDtype::F32)
            || node.kind != PhysicalLaunchKind::Gemm
            || node.symbol != arm.symbol
            || node.module_kind != ModuleKind::TriadSm80
            || node.logical_op != ResolvedGemmOp::Nt
            || node.logical_dtype != PolicyDtype::F32
            || node.execution_dtype != PolicyDtype::F32
            || node.shape != cell.dims
            || node.strides != expected_strides
            || node.tile != Some(arm.tile)
            || node.launch.grid_dim != expected.grid_dim
            || node.launch.block_dim != expected.block_dim
            || node.launch.shared_mem_bytes != expected.shared_mem_bytes
        {
            return Err(format!(
                "{} {} physical manifest changed: {:?}",
                cell.name,
                arm.name,
                evidence.nodes()
            ));
        }
        Ok(())
    }

    fn measure(
        launch: &mut QualifiedPhysicalLaunch<'_>,
        ctx: &GpuCtx,
        path: PathKind,
        iterations: usize,
    ) -> Result<f64, String> {
        let total_ms = match path {
            PathKind::Eager => launch.measure_eager_window_ms(ctx, iterations)?,
            PathKind::Graph => launch.measure_graph_window_ms(ctx, iterations)?,
        };
        let us = total_ms * 1_000.0 / iterations as f64;
        require_positive_finite("timing window", us)?;
        Ok(us)
    }

    fn require_positive_finite(label: &str, value: f64) -> Result<(), String> {
        if value.is_finite() && value > 0.0 {
            Ok(())
        } else {
            Err(format!("{label} is not positive and finite: {value}"))
        }
    }

    fn output_bits(launch: &QualifiedPhysicalLaunch<'_>, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
        launch.f32_output_bits(ctx)
    }

    fn output_values(
        launch: &QualifiedPhysicalLaunch<'_>,
        ctx: &GpuCtx,
    ) -> Result<Vec<f32>, String> {
        Ok(output_bits(launch, ctx)?
            .into_iter()
            .map(f32::from_bits)
            .collect())
    }

    fn correctness_gate<'ctx>(
        cell: Cell,
        contexts: &'ctx [GpuCtx],
        launches: &mut [QualifiedPhysicalLaunch<'ctx>],
    ) -> Result<CorrectnessEvidence, String> {
        if contexts.len() != ARMS.len() || launches.len() != ARMS.len() {
            return Err("TF32 NT correctness gate requires exactly four arms".into());
        }
        let oracle = nt_reference(cell.dims, SALT)?;
        let oracle_bits = oracle
            .iter()
            .map(|value| (*value as f32).to_bits())
            .collect::<Vec<_>>();
        let mut references: [Option<Vec<u32>>; 4] = std::array::from_fn(|_| None);
        for (index, (ctx, launch)) in contexts.iter().zip(launches.iter_mut()).enumerate() {
            for path in [PathKind::Eager, PathKind::Graph] {
                for repeat in 0..DETERMINISM_REPEATS {
                    launch.seed_f32_operands(ctx, SALT)?;
                    measure(launch, ctx, path, 1)?;
                    let actual_values = output_values(launch, ctx)?;
                    check_numeric(
                        &actual_values,
                        &oracle,
                        &format!("{} {} {}", cell.name, ARMS[index].name, path.name()),
                    )?;
                    let actual = actual_values
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>();
                    if references[index]
                        .as_ref()
                        .is_some_and(|reference| reference != &actual)
                    {
                        return Err(format!(
                            "{} {} changed output bits on {} repeat {repeat}",
                            cell.name,
                            ARMS[index].name,
                            path.name()
                        ));
                    }
                    references[index].get_or_insert(actual);
                }
            }
        }
        let reference_bits = references.map(|reference| {
            reference.expect("every TF32 NT arm ran eager and graph correctness gates")
        });
        Ok(CorrectnessEvidence {
            output_digests: std::array::from_fn(|index| bits_digest(&reference_bits[index])),
            oracle_digest: bits_digest(&oracle_bits),
            reference_bits,
        })
    }

    fn calibrate(
        launch: &mut QualifiedPhysicalLaunch<'_>,
        ctx: &GpuCtx,
        path: PathKind,
    ) -> Result<usize, String> {
        let pilot_us = measure(launch, ctx, path, PILOT_ITERATIONS)?;
        Ok(((TARGET_WINDOW_US / pilot_us).ceil() as usize).clamp(1, MAX_WINDOW_ITERATIONS))
    }

    fn measure_order<'ctx>(
        contexts: &'ctx [GpuCtx],
        launches: &mut [QualifiedPhysicalLaunch<'ctx>],
        path: PathKind,
        order: &[usize],
        iterations: &[usize; 4],
    ) -> Result<[f64; 4], String> {
        let mut values = [0.0; 4];
        for &index in order {
            values[index] = measure(
                &mut launches[index],
                &contexts[index],
                path,
                iterations[index],
            )?;
        }
        Ok(values)
    }

    fn percentiles(values: &[f64], windows: usize) -> Result<Percentiles, String> {
        if values.len() != windows || windows < 3 {
            return Err(format!(
                "TF32 NT percentile requires {windows} samples, received {}",
                values.len()
            ));
        }
        let mut ordered = values.to_vec();
        ordered.sort_by(f64::total_cmp);
        let rank = |percent: usize| (ordered.len() - 1) * percent / 100;
        Ok(Percentiles {
            p05: ordered[rank(5)],
            p50: ordered[rank(50)],
            p95: ordered[rank(95)],
        })
    }

    fn summarize(samples: &TournamentSamples, windows: usize) -> Result<TournamentSummary, String> {
        let summarize_order = |order: &OrderedSamples| {
            Ok::<_, String>([
                percentiles(&order.arm_us[0], windows)?,
                percentiles(&order.arm_us[1], windows)?,
                percentiles(&order.arm_us[2], windows)?,
                percentiles(&order.arm_us[3], windows)?,
            ])
        };
        let forward = summarize_order(&samples.forward)?;
        let reverse = summarize_order(&samples.reverse)?;
        let fastest = |values: &[Percentiles; 4]| {
            values
                .iter()
                .enumerate()
                .min_by(|left, right| left.1.p50.total_cmp(&right.1.p50))
                .map(|(index, _)| index)
        };
        let forward_winner = fastest(&forward);
        let reverse_winner = fastest(&reverse);
        Ok(TournamentSummary {
            forward,
            reverse,
            measured_winner: (forward_winner == reverse_winner)
                .then_some(forward_winner)
                .flatten(),
        })
    }

    fn run_cohort(
        run: CohortRun<'_, '_>,
    ) -> Result<(TournamentSamples, TournamentSummary, CohortEvidence), String> {
        let label = format!("tf32-nt/{}/{}", run.cell.name, run.path.name());
        let calibration_preflight = run.quiet.require_cohort(&format!("{label}/calibration"))?;
        for (ctx, launch) in run.contexts.iter().zip(run.launches.iter_mut()) {
            measure(launch, ctx, run.path, WARMUPS)?;
        }
        let mut iterations = [0usize; 4];
        for (index, iteration) in iterations.iter_mut().enumerate() {
            *iteration = calibrate(&mut run.launches[index], &run.contexts[index], run.path)?;
        }
        let timed_preflight = run.quiet.require_cohort(&format!("{label}/timed"))?;
        let mut samples = TournamentSamples {
            forward: OrderedSamples::with_capacity(run.windows),
            reverse: OrderedSamples::with_capacity(run.windows),
        };
        for window in 0..run.windows {
            let forward = measure_order(
                run.contexts,
                run.launches,
                run.path,
                &latin_order(window, false),
                &iterations,
            )?;
            samples.forward.push(forward)?;
            let reverse = measure_order(
                run.contexts,
                run.launches,
                run.path,
                &latin_order(window, true),
                &iterations,
            )?;
            samples.reverse.push(reverse)?;
        }
        let postflight = run.quiet.verify_post_cohort(&format!("{label}/post"))?;
        for (index, arm) in ARMS.iter().enumerate() {
            run.launches[index].seed_f32_operands(&run.contexts[index], SALT)?;
            measure(&mut run.launches[index], &run.contexts[index], run.path, 1)?;
            if output_bits(&run.launches[index], &run.contexts[index])?
                != run.correctness.reference_bits[index]
            {
                return Err(format!(
                    "{} {} changed bits after the {} timing cohort",
                    run.cell.name,
                    arm.name,
                    run.path.name()
                ));
            }
        }
        let summary = summarize(&samples, run.windows)?;
        Ok((
            samples,
            summary,
            CohortEvidence {
                calibration_preflight,
                timed_preflight,
                postflight,
                iterations,
            },
        ))
    }

    fn render_measurement(record: MeasurementRecord<'_>) -> String {
        let metadata = record.metadata;
        let (m, k, n) = record.cell.dims;
        let winner = record
            .summary
            .measured_winner
            .map(|index| ARMS[index].name)
            .unwrap_or("inconclusive");
        let mut json = String::new();
        write!(
            json,
            concat!(
                "{{\"schema\":\"{}\",\"record_type\":\"route_measurement\",",
                "\"run_identity\":\"{}\",\"cell\":\"{}\",",
                "\"m\":{},\"k\":{},\"n\":{},\"path\":\"{}\",",
                "\"windows_per_order\":{},",
                "\"order_policy\":\"latin_rotation_with_per_window_reverse\",",
                "\"measurement_only\":true,\"runtime_autotuning\":false,",
                "\"dispatcher_admitted\":false,",
                "\"decision\":\"{}\",\"measured_winner\":\"{}\",",
                "\"gpu_uuid\":\"{}\",\"cc\":\"{}.{}\",\"multiprocessors\":{},",
                "\"driver_api_version\":{},\"driver_build_digest\":\"{}\",",
                "\"nvrtc_version\":\"{}.{}\",\"nvrtc_target\":\"{}\",",
                "\"sm80_compile_key\":\"{}\",\"sm80_artifact_digest\":\"{}\",",
                "\"sm80_invocation_digest\":\"{}\",",
                "\"tournament_source_sha256\":\"{}\",\"cuda_source_sha256\":\"{}\",",
                "\"pre_context\":\"{}\",\"calibration_preflight\":\"{}\",",
                "\"timed_preflight\":\"{}\",\"postflight\":\"{}\",",
                "\"correctness_oracle\":\"cpu_tf32_rna_nt_f64_accumulation\",",
                "\"oracle_tolerance\":\"0.0025*(1+abs(reference))\",",
                "\"oracle_digest\":\"{}\",\"route_self_repeat_bit_exact\":true,",
                "\"eager_graph_bit_exact\":true,\"determinism_repeats_per_path\":{},",
                "\"resource_gate_enforced_by_loader\":true,\"arms\":["
            ),
            SCHEMA,
            metadata.run_identity,
            record.cell.name,
            m,
            k,
            n,
            record.path.name(),
            record.windows,
            DECISION,
            winner,
            escape_json(&metadata.gpu_uuid),
            metadata.compute_capability.0,
            metadata.compute_capability.1,
            metadata.multiprocessors,
            metadata.driver_api_version,
            metadata.driver_build_digest,
            metadata.nvrtc_version.0,
            metadata.nvrtc_version.1,
            escape_json(&metadata.nvrtc_target),
            metadata.compile_key,
            metadata.artifact_digest,
            metadata.invocation_digest,
            metadata.tournament_source_sha256,
            metadata.cuda_source_sha256,
            escape_json(&metadata.pre_context),
            escape_json(&record.cohort.calibration_preflight),
            escape_json(&record.cohort.timed_preflight),
            escape_json(&record.cohort.postflight),
            hex_digest(&record.correctness.oracle_digest),
            DETERMINISM_REPEATS,
        )
        .expect("String writes cannot fail");
        for (index, arm) in ARMS.iter().copied().enumerate() {
            if index != 0 {
                json.push(',');
            }
            let evidence = record.launches[index].evidence();
            let manifest = expected_manifest(record.cell, arm);
            write!(
                json,
                concat!(
                    "{{\"index\":{},\"name\":\"{}\",\"symbol\":\"{}\",",
                    "\"tile\":[{},{}],\"bk\":32,\"partitions\":{},\"stages\":{},",
                    "\"threads\":128,\"dynamic_shared_bytes\":{},\"register_cap\":{},",
                    "\"occupancy_gate\":{},",
                    "\"grid\":[{},{},{}],\"block\":[128,1,1],\"node_count\":1,",
                    "\"launch_digest\":\"{}\",\"request_digest\":\"{}\",",
                    "\"output_digest\":\"{}\",\"iterations\":{},",
                    "\"forward_us\":{{\"p05\":{:.9},\"p50\":{:.9},\"p95\":{:.9}}},",
                    "\"reverse_us\":{{\"p05\":{:.9},\"p50\":{:.9},\"p95\":{:.9}}},",
                    "\"forward_samples_us\":[{}],\"reverse_samples_us\":[{}]}}"
                ),
                index,
                arm.name,
                arm.symbol,
                arm.tile.0,
                arm.tile.1,
                arm.partitions,
                arm.stages,
                arm.dynamic_shared_bytes,
                arm.register_cap,
                arm.occupancy_gate,
                manifest.grid_dim.0,
                manifest.grid_dim.1,
                manifest.grid_dim.2,
                hex_digest(&evidence.launch_digest()),
                hex_digest(&evidence.request_identity_digest()),
                hex_digest(&record.correctness.output_digests[index]),
                record.cohort.iterations[index],
                record.summary.forward[index].p05,
                record.summary.forward[index].p50,
                record.summary.forward[index].p95,
                record.summary.reverse[index].p05,
                record.summary.reverse[index].p50,
                record.summary.reverse[index].p95,
                render_samples(&record.samples.forward.arm_us[index]),
                render_samples(&record.samples.reverse.arm_us[index]),
            )
            .expect("String writes cannot fail");
        }
        json.push_str("]}");
        json
    }

    fn render_samples(values: &[f64]) -> String {
        let mut rendered = String::new();
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                rendered.push(',');
            }
            write!(rendered, "{value:.9}").expect("String writes cannot fail");
        }
        rendered
    }

    fn bits_digest(bits: &[u32]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"gemm-bi-tf32-nt-output-bits.v1");
        for value in bits {
            digest.update(value.to_le_bytes());
        }
        digest.finalize().into()
    }

    fn sha256_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn hex_digest(bytes: &[u8; 32]) -> String {
        let mut rendered = String::with_capacity(64);
        for byte in bytes {
            write!(rendered, "{byte:02x}").expect("String writes cannot fail");
        }
        rendered
    }

    fn escape_json(value: &str) -> String {
        let mut escaped = String::with_capacity(value.len());
        for character in value.chars() {
            match character {
                '"' => escaped.push_str("\\\""),
                '\\' => escaped.push_str("\\\\"),
                '\n' => escaped.push_str("\\n"),
                '\r' => escaped.push_str("\\r"),
                '\t' => escaped.push_str("\\t"),
                character if character.is_control() => {
                    write!(escaped, "\\u{:04x}", character as u32)
                        .expect("String writes cannot fail");
                }
                character => escaped.push(character),
            }
        }
        escaped
    }

    fn presize_contexts(contexts: &[GpuCtx]) -> Result<(), String> {
        for (index, ctx) in contexts.iter().enumerate() {
            let requests = CELLS.map(|cell| request(cell, ARMS[index]));
            presize_physical_qualification_suite(ctx, &requests)?;
        }
        Ok(())
    }

    fn qualify_cell<'ctx>(
        contexts: &'ctx [GpuCtx],
        cell: Cell,
    ) -> Result<Vec<QualifiedPhysicalLaunch<'ctx>>, String> {
        contexts
            .iter()
            .zip(ARMS)
            .map(|(ctx, arm)| qualify_physical_launch(ctx, request(cell, arm)))
            .collect()
    }

    #[test]
    #[ignore = "requires a quiet SM80+ CUDA device, NVRTC, and emits performance JSONL"]
    fn forced_tf32_nt_route_tournament_is_measurement_only() {
        validate_release_build().unwrap();
        let windows = window_count(std::env::var(FINAL_ENV).ok().as_deref()).unwrap();
        let quiet = QuietGpu::for_cuda_ordinal(0).unwrap();
        let pre_context = quiet.require_pre_context("tf32-nt/pre-context").unwrap();
        let device = GpuDevice::new(0).unwrap();
        validate_device(&device).unwrap();
        let contexts = (0..ARMS.len())
            .map(|_| context(&device))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        presize_contexts(&contexts).unwrap();
        let mut metadata = None;
        let mut pre_context = Some(pre_context);
        let mut sink = JsonlSink::create_from_env().unwrap();

        for cell in CELLS {
            let mut launches = qualify_cell(&contexts, cell).unwrap();
            let metadata = metadata.get_or_insert_with(|| {
                RunMetadata::new(&launches[0], &device, &quiet, pre_context.take().unwrap())
            });
            for (arm, launch) in ARMS.into_iter().zip(&launches) {
                assert_manifest(cell, arm, launch).unwrap();
            }
            let correctness = correctness_gate(cell, &contexts, &mut launches).unwrap();
            for path in [PathKind::Eager, PathKind::Graph] {
                let (samples, summary, cohort) = run_cohort(CohortRun {
                    cell,
                    path,
                    quiet: &quiet,
                    contexts: &contexts,
                    launches: &mut launches,
                    correctness: &correctness,
                    windows,
                })
                .unwrap();
                sink.write(render_measurement(MeasurementRecord {
                    metadata,
                    cell,
                    path,
                    windows,
                    launches: &launches,
                    correctness: &correctness,
                    samples: &samples,
                    summary: &summary,
                    cohort: &cohort,
                }))
                .unwrap();
            }
        }
        sink.finish(windows).unwrap();
    }

    #[test]
    fn cpu_tf32_nt_oracle_matches_the_seed_contract_and_rejects_corruption() {
        let dims = (2, 3, 4);
        let salt = 0x6e74_7038;
        let expected = nt_reference(dims, salt).unwrap();
        let a = seeded_values(dims.0 * dims.2, salt ^ 0x2d);
        let b = seeded_values(dims.1 * dims.2, salt ^ 0x67);
        let row = 1;
        let column = 2;
        let manual = (0..dims.2)
            .map(|inner| {
                tf32_value(a[row * dims.2 + inner]) * tf32_value(b[column * dims.2 + inner])
            })
            .sum::<f64>();
        assert_eq!(expected[row * dims.1 + column], manual);

        let mut actual = expected
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        check_numeric(&actual, &expected, "oracle").unwrap();
        actual[row * dims.1 + column] = f32::INFINITY;
        assert!(check_numeric(&actual, &expected, "oracle").is_err());
    }
}
