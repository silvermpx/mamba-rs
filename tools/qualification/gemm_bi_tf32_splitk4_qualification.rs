#![cfg(feature = "cuda")]

#[path = "../../tests/common/gpu_quiet.rs"]
mod gpu_quiet;

use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write as _};
use std::path::PathBuf;

use gpu_quiet::QuietGpu;
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    PhysicalQualificationRequest, PhysicalQualificationRoute, QualifiedPhysicalLaunch,
    Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile,
    presize_physical_qualification_suite, qualify_physical_launch,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;
use sha2::{Digest as _, Sha256};

const OUTPUT_ENV: &str = "MAMBA_RS_TF32_SPLITK4_JSONL";
const WINDOWS: usize = 101;
const WARMUPS: usize = 128;
const DETERMINISM_REPEATS: usize = 10;
const TARGET_WINDOW_MS: f64 = 5.0;
const S4_FUSED_SYMBOL: &str = "gemm_bi_nn_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4";
const DIRECT_SYMBOL: &str = "gemm_bi_nn_sm80_mma_tf32_v1_m16n32_bk32_s4";

#[derive(Clone, Copy)]
struct Cell {
    name: &'static str,
    dims: (usize, usize, usize),
}

const CELLS: [Cell; 2] = [
    Cell {
        name: "m64_k1536_n384",
        dims: (64, 1_536, 384),
    },
    Cell {
        name: "m64_k833_n384",
        dims: (64, 833, 384),
    },
];

#[derive(Clone, Copy)]
enum Path {
    Eager,
    Graph,
}

impl Path {
    const fn name(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Graph => "graph",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    SplitS4,
    DirectS4,
}

impl Arm {
    const fn name(self) -> &'static str {
        match self {
            Self::SplitS4 => "splitk4_s4",
            Self::DirectS4 => "direct_s4",
        }
    }

    const fn route(self) -> Tf32PhysicalRoute {
        let portable = Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        };
        match self {
            Self::SplitS4 => Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(portable),
            Self::DirectS4 => Tf32PhysicalRoute::MmaTf32RnaV1(portable),
        }
    }

    const fn fused_symbol(self) -> Option<&'static str> {
        match self {
            Self::SplitS4 => Some(S4_FUSED_SYMBOL),
            Self::DirectS4 => None,
        }
    }

    const fn dynamic_shared_bytes(self) -> Option<u32> {
        match self {
            Self::SplitS4 => Some(29_696),
            Self::DirectS4 => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Comparison {
    suite: &'static str,
    left: Arm,
    right: Arm,
    require_dense_cross_exact: bool,
}

const SPLIT_DIRECT: Comparison = Comparison {
    suite: "gemm_bi_tf32_splitk4_vs_direct",
    left: Arm::SplitS4,
    right: Arm::DirectS4,
    require_dense_cross_exact: false,
};

const DIRECT_SPLIT: Comparison = Comparison {
    suite: "gemm_bi_tf32_direct_vs_splitk4_swapped_contexts",
    left: Arm::DirectS4,
    right: Arm::SplitS4,
    require_dense_cross_exact: false,
};

struct PairRuntime<'a> {
    left_ctx: &'a GpuCtx,
    right_ctx: &'a GpuCtx,
    quiet_gpu: &'a QuietGpu,
    assignment_preflight_snapshot: &'a str,
    multiprocessors: u32,
}

struct CorrectnessEvidence {
    semantic_output_digest: [u8; 32],
    left_dense_output_digest: [u8; 32],
    right_dense_output_digest: [u8; 32],
    dense_cross_route_exact: bool,
}

struct CellRuntime<'a> {
    pair: &'a PairRuntime<'a>,
    correctness: &'a CorrectnessEvidence,
}

struct OrderedSamples {
    left_us: Vec<f64>,
    right_us: Vec<f64>,
    ratios: Vec<f64>,
}

impl OrderedSamples {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            left_us: Vec::with_capacity(capacity),
            right_us: Vec::with_capacity(capacity),
            ratios: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, left_us: f64, right_us: f64) -> Result<(), String> {
        validate_sample("left sample", left_us)?;
        validate_sample("right sample", right_us)?;
        let ratio = left_us / right_us;
        validate_sample("paired ratio", ratio)?;
        self.left_us.push(left_us);
        self.right_us.push(right_us);
        self.ratios.push(ratio);
        Ok(())
    }
}

struct PairSamples {
    left_first: OrderedSamples,
    right_first: OrderedSamples,
}

impl PairSamples {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            left_first: OrderedSamples::with_capacity(capacity),
            right_first: OrderedSamples::with_capacity(capacity),
        }
    }
}

#[derive(Clone, Copy)]
struct RatioSummary {
    left_first_p05: f64,
    left_first_p50: f64,
    left_first_p95: f64,
    right_first_p05: f64,
    right_first_p50: f64,
    right_first_p95: f64,
    left_first_median_low: f64,
    left_first_median_high: f64,
    right_first_median_low: f64,
    right_first_median_high: f64,
}

#[derive(Clone, Copy)]
struct CohortDecision {
    comparison: &'static str,
    cell: &'static str,
    path: &'static str,
    retained: &'static str,
}

struct CohortEvidence {
    calibration_preflight: String,
    timed_preflight: String,
    postflight: String,
    left_iterations: usize,
    right_iterations: usize,
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
            .map_err(|error| format!("create split-K4 evidence {path:?}: {error}"))?;
        if !file
            .metadata()
            .map_err(|error| format!("inspect split-K4 evidence {path:?}: {error}"))?
            .file_type()
            .is_file()
        {
            return Err(format!("split-K4 evidence {path:?} is not a regular file"));
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
            .map_err(|error| format!("write split-K4 evidence {:?}: {error}", self.path))?;
        self.digest.update(&bytes);
        self.records += 1;
        Ok(())
    }

    fn finish(
        mut self,
        expected_records: usize,
        decision: &str,
        quiet_gpu: &QuietGpu,
        device: &GpuDevice,
    ) -> Result<(), String> {
        if self.records != expected_records {
            return Err(format!(
                "incomplete split-K4 evidence: got {} records, expected {expected_records}",
                self.records
            ));
        }
        let prior_digest = format!("{:x}", self.digest.clone().finalize());
        self.write(format!(
            "{{\"schema\":\"MambaBiTf32SplitK4CompletionV1\",\"cohort_valid\":true,\"gpu_uuid\":\"{}\",\"cc\":\"{}.{}\",\"multiprocessors\":{},\"cohort_records\":{},\"cohort_digest\":\"{}\",\"decision\":\"{}\"}}",
            quiet_gpu.uuid,
            device.compute_capability.0,
            device.compute_capability.1,
            device.multiprocessor_count(),
            expected_records,
            prior_digest,
            escape_json_string(decision),
        ))?;
        self.writer
            .flush()
            .map_err(|error| format!("flush split-K4 evidence {:?}: {error}", self.path))?;
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("sync split-K4 evidence {:?}: {error}", self.path))?;
        eprintln!(
            "split-K4 evidence complete: {:?}; records={}; digest={prior_digest}",
            self.path, self.records
        );
        Ok(())
    }
}

fn context(device: &GpuDevice) -> Result<GpuCtx, String> {
    let ctx = GpuCtx::new(device)?;
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    Ok(ctx)
}

fn request(cell: Cell, arm: Arm) -> PhysicalQualificationRequest {
    PhysicalQualificationRequest::contiguous(
        ResolvedGemmOp::Nn,
        cell.dims,
        PhysicalQualificationRoute::Tf32Forced(arm.route()),
    )
}

fn measure(
    launch: &mut QualifiedPhysicalLaunch<'_>,
    ctx: &GpuCtx,
    path: Path,
    iterations: usize,
) -> Result<f64, String> {
    let total = match path {
        Path::Eager => launch.measure_eager_window_ms(ctx, iterations)?,
        Path::Graph => launch.measure_graph_window_ms(ctx, iterations)?,
    };
    Ok(total * 1_000.0 / iterations as f64)
}

fn calibrate(
    launch: &mut QualifiedPhysicalLaunch<'_>,
    ctx: &GpuCtx,
    path: Path,
) -> Result<usize, String> {
    let pilot = measure(launch, ctx, path, 16)?;
    if !pilot.is_finite() || pilot <= 0.0 {
        return Err(format!("{} pilot is not positive and finite", path.name()));
    }
    Ok(((TARGET_WINDOW_MS * 1_000.0 / pilot).ceil() as usize).clamp(1, 1_000_000))
}

fn percentile(values: &[f64], index: usize) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[index]
}

fn summarize(samples: &PairSamples) -> Result<RatioSummary, String> {
    if samples.left_first.ratios.len() != WINDOWS || samples.right_first.ratios.len() != WINDOWS {
        return Err(format!(
            "the performance set requires exactly {WINDOWS} paired windows"
        ));
    }
    let summarize_order = |values: &[f64]| {
        (
            percentile(values, 5),
            percentile(values, 40),
            percentile(values, 50),
            percentile(values, 60),
            percentile(values, 95),
        )
    };
    let (left_p05, left_low, left_p50, left_high, left_p95) =
        summarize_order(&samples.left_first.ratios);
    let (right_p05, right_low, right_p50, right_high, right_p95) =
        summarize_order(&samples.right_first.ratios);
    Ok(RatioSummary {
        left_first_p05: left_p05,
        left_first_p50: left_p50,
        left_first_p95: left_p95,
        right_first_p05: right_p05,
        right_first_p50: right_p50,
        right_first_p95: right_p95,
        left_first_median_low: left_low,
        left_first_median_high: left_high,
        right_first_median_low: right_low,
        right_first_median_high: right_high,
    })
}

fn validate_sample(label: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!("{label} was not positive and finite: {value}"))
    }
}

fn validate_release_build(debug_assertions: bool) -> Result<(), String> {
    if debug_assertions {
        Err("split-K4 performance qualification requires --release".into())
    } else {
        Ok(())
    }
}

fn validate_device(device: &GpuDevice) -> Result<(), String> {
    if matches!(device.compute_capability, (8, 9) | (12, 0)) {
        Ok(())
    } else {
        Err(format!(
            "split-K4 performance acceptance is qualified only on exact SM89 or SM120, received SM{}.{}",
            device.compute_capability.0, device.compute_capability.1
        ))
    }
}

fn assert_manifest(
    cell: Cell,
    arm: Arm,
    launch: &QualifiedPhysicalLaunch<'_>,
) -> Result<(), String> {
    let evidence = launch.evidence();
    if !evidence.eager_graph_equal() {
        return Err(format!(
            "{} {} eager/graph manifest differs",
            cell.name,
            arm.name()
        ));
    }
    match (arm.fused_symbol(), arm.dynamic_shared_bytes()) {
        (Some(fused_symbol), Some(dynamic_shared_bytes)) => {
            let symbols = evidence
                .nodes()
                .iter()
                .map(|node| node.symbol)
                .collect::<Vec<_>>();
            if evidence.launch_count() != 1
                || evidence.single_launch_symbol() != Some(fused_symbol)
                || evidence.single_launch_tile() != Some((16, 32))
                || symbols != [fused_symbol]
                || evidence.nodes()[0].tile != Some((16, 32))
                || evidence.nodes()[0].launch.grid_dim != (12, 4, 4)
                || evidence.nodes()[0].launch.block_dim != (128, 1, 1)
                || evidence.nodes()[0].launch.shared_mem_bytes != dynamic_shared_bytes
            {
                return Err(format!(
                    "{} {} physical manifest changed: {:?}",
                    cell.name,
                    arm.name(),
                    evidence.nodes()
                ));
            }
        }
        (None, None) => {
            if evidence.launch_count() != 1
                || evidence.single_launch_symbol() != Some(DIRECT_SYMBOL)
                || evidence.single_launch_tile() != Some((16, 32))
                || evidence.nodes()[0].tile != Some((16, 32))
                || evidence.nodes()[0].launch.grid_dim != (48, 1, 1)
                || evidence.nodes()[0].launch.block_dim != (128, 1, 1)
                || evidence.nodes()[0].launch.shared_mem_bytes != 29_696
            {
                return Err(format!(
                    "{} direct M16N32 physical manifest changed: {:?}",
                    cell.name,
                    evidence.nodes()
                ));
            }
        }
        _ => {
            return Err(format!(
                "{} has an incomplete manifest contract",
                arm.name()
            ));
        }
    }
    Ok(())
}

fn launch_once(
    launch: &mut QualifiedPhysicalLaunch<'_>,
    ctx: &GpuCtx,
    path: Path,
) -> Result<(), String> {
    measure(launch, ctx, path, 1).map(|_| ())
}

fn semantic_gate(
    cell: Cell,
    left: &mut QualifiedPhysicalLaunch<'_>,
    right: &mut QualifiedPhysicalLaunch<'_>,
    runtime: &PairRuntime<'_>,
) -> Result<[u8; 32], String> {
    let (m, _, n) = cell.dims;
    let mut semantic_reference = None;
    for path in [Path::Eager, Path::Graph] {
        left.seed_f32_nn_single_term_probe(runtime.left_ctx)?;
        right.seed_f32_nn_single_term_probe(runtime.right_ctx)?;
        launch_once(left, runtime.left_ctx, path)?;
        launch_once(right, runtime.right_ctx, path)?;
        let left_bits = left.f32_output_bits(runtime.left_ctx)?;
        let right_bits = right.f32_output_bits(runtime.right_ctx)?;
        if left_bits.len() != m * n || right_bits.len() != m * n {
            return Err(format!(
                "{} one-term probe returned the wrong output extent",
                cell.name
            ));
        }
        for row in 0..m {
            let lhs = ((row % 7) as i32 - 3) as f32 * 0.125;
            for column in 0..n {
                let rhs = ((column % 11) as i32 - 5) as f32 * 0.125;
                let expected = lhs.mul_add(rhs, 0.0).to_bits();
                let index = row * n + column;
                if left_bits[index] != expected || right_bits[index] != expected {
                    return Err(format!(
                        "{} {} one-term probe differs at ({row},{column}): left={:#010x}, right={:#010x}, expected={expected:#010x}",
                        cell.name,
                        path.name(),
                        left_bits[index],
                        right_bits[index]
                    ));
                }
            }
        }
        semantic_reference.get_or_insert(left_bits);
    }
    semantic_reference
        .as_deref()
        .map(output_bits_digest)
        .ok_or_else(|| format!("{} semantic gate produced no output", cell.name))
}

fn dense_determinism_gate(
    cell: Cell,
    comparison: Comparison,
    left: &mut QualifiedPhysicalLaunch<'_>,
    right: &mut QualifiedPhysicalLaunch<'_>,
    runtime: &PairRuntime<'_>,
    semantic_output_digest: [u8; 32],
) -> Result<CorrectnessEvidence, String> {
    let mut left_reference = None;
    let mut right_reference = None;
    for path in [Path::Eager, Path::Graph] {
        for _ in 0..DETERMINISM_REPEATS {
            left.seed_f32_operands(runtime.left_ctx, 0x7b31_4d29)?;
            right.seed_f32_operands(runtime.right_ctx, 0x7b31_4d29)?;
            launch_once(left, runtime.left_ctx, path)?;
            launch_once(right, runtime.right_ctx, path)?;
            let left_actual = left.f32_output_bits(runtime.left_ctx)?;
            let right_actual = right.f32_output_bits(runtime.right_ctx)?;
            if left_reference
                .as_ref()
                .is_some_and(|bits| bits != &left_actual)
                || right_reference
                    .as_ref()
                    .is_some_and(|bits| bits != &right_actual)
            {
                return Err(format!(
                    "{} eager or graph output bits changed across repeats",
                    cell.name
                ));
            }
            if comparison.require_dense_cross_exact && left_actual != right_actual {
                return Err(format!(
                    "{} {} and {} changed the split-K4 numeric family",
                    cell.name,
                    comparison.left.name(),
                    comparison.right.name()
                ));
            }
            left_reference.get_or_insert(left_actual);
            right_reference.get_or_insert(right_actual);
        }
    }
    let left_reference = left_reference
        .ok_or_else(|| format!("{} left dense gate produced no output", cell.name))?;
    let right_reference = right_reference
        .ok_or_else(|| format!("{} right dense gate produced no output", cell.name))?;
    Ok(CorrectnessEvidence {
        semantic_output_digest,
        left_dense_output_digest: output_bits_digest(&left_reference),
        right_dense_output_digest: output_bits_digest(&right_reference),
        dense_cross_route_exact: left_reference == right_reference,
    })
}

fn output_bits_digest(bits: &[u32]) -> [u8; 32] {
    let mut digest = Sha256::new();
    for bit in bits {
        digest.update(bit.to_le_bytes());
    }
    digest.finalize().into()
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

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut rendered = String::with_capacity(64);
    for byte in bytes {
        write!(rendered, "{byte:02x}").expect("String writes cannot fail");
    }
    rendered
}

fn escape_json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                write!(escaped, "\\u{:04x}", character as u32).expect("String writes cannot fail");
            }
            character => escaped.push(character),
        }
    }
    escaped
}

fn retention_decision(comparison: Comparison, summary: RatioSummary) -> &'static str {
    let left_wins = summary.left_first_p50 < 1.0
        && summary.left_first_p95 < 1.0
        && summary.right_first_p50 < 1.0
        && summary.right_first_p95 < 1.0;
    let right_wins = summary.left_first_p05 > 1.0
        && summary.left_first_p50 > 1.0
        && summary.right_first_p05 > 1.0
        && summary.right_first_p50 > 1.0;
    if left_wins {
        comparison.left.name()
    } else if right_wins {
        comparison.right.name()
    } else {
        "inconclusive"
    }
}

struct EvidenceRecord<'a> {
    cell: Cell,
    path: Path,
    comparison: Comparison,
    runtime: &'a CellRuntime<'a>,
    samples: &'a PairSamples,
    summary: RatioSummary,
    cohort: &'a CohortEvidence,
}

fn emit_evidence(
    record: EvidenceRecord<'_>,
    left: &QualifiedPhysicalLaunch<'_>,
    right: &QualifiedPhysicalLaunch<'_>,
) -> String {
    let cc = record
        .runtime
        .pair
        .left_ctx
        .stream
        .context()
        .compute_capability()
        .expect("split-K4 CUDA CC");
    let left_evidence = left.evidence();
    let right_evidence = right.evidence();
    let (m, k, n) = record.cell.dims;
    format!(
        concat!(
            "{{\"schema\":\"MambaBiTf32SplitK4PairedPerformanceV1\",",
            "\"suite\":\"{}\",\"cohort_valid\":true,",
            "\"cell\":\"{}\",\"m\":{},\"k\":{},\"n\":{},",
            "\"path\":\"{}\",\"cc\":\"{}.{}\",\"multiprocessors\":{},",
            "\"gpu_uuid\":\"{}\",",
            "\"left\":\"{}\",\"right\":\"{}\",",
            "\"semantic_probe\":\"one_active_term_eager_graph_host_exact\",",
            "\"semantic_eager_host_exact\":true,\"semantic_graph_host_exact\":true,",
            "\"dense_eager_repeat_exact\":true,\"dense_graph_repeat_exact\":true,",
            "\"dense_eager_graph_exact\":true,\"dense_cross_route_exact_required\":{},",
            "\"dense_cross_route_exact\":{},\"semantic_output_digest\":\"{}\",",
            "\"left_dense_output_digest\":\"{}\",\"right_dense_output_digest\":\"{}\",",
            "\"determinism_repeats_per_path\":{},\"windows\":{},",
            "\"assignment_preflight\":\"{}\",\"calibration_preflight\":\"{}\",",
            "\"timed_preflight\":\"{}\",\"postflight\":\"{}\",",
            "\"left_launch_digest\":\"{}\",\"right_launch_digest\":\"{}\",",
            "\"left_request_digest\":\"{}\",\"right_request_digest\":\"{}\",",
            "\"left_route\":\"{}\",\"right_route\":\"{}\",",
            "\"left_iterations\":{},\"right_iterations\":{},",
            "\"left_first_p05\":{:.9},\"left_first_p50\":{:.9},",
            "\"left_first_p95\":{:.9},\"right_first_p05\":{:.9},",
            "\"right_first_p50\":{:.9},\"right_first_p95\":{:.9},",
            "\"left_first_median_interval\":[{:.9},{:.9}],",
            "\"right_first_median_interval\":[{:.9},{:.9}],",
            "\"decision\":\"{}\",",
            "\"left_manifest\":\"{}\",\"right_manifest\":\"{}\",",
            "\"left_first_left_us\":[{}],\"left_first_right_us\":[{}],",
            "\"left_first_ratios\":[{}],\"right_first_left_us\":[{}],",
            "\"right_first_right_us\":[{}],\"right_first_ratios\":[{}]}}"
        ),
        record.comparison.suite,
        record.cell.name,
        m,
        k,
        n,
        record.path.name(),
        cc.0,
        cc.1,
        record.runtime.pair.multiprocessors,
        record.runtime.pair.quiet_gpu.uuid,
        record.comparison.left.name(),
        record.comparison.right.name(),
        record.comparison.require_dense_cross_exact,
        record.runtime.correctness.dense_cross_route_exact,
        hex_digest(&record.runtime.correctness.semantic_output_digest),
        hex_digest(&record.runtime.correctness.left_dense_output_digest),
        hex_digest(&record.runtime.correctness.right_dense_output_digest),
        DETERMINISM_REPEATS,
        WINDOWS,
        escape_json_string(record.runtime.pair.assignment_preflight_snapshot),
        escape_json_string(&record.cohort.calibration_preflight),
        escape_json_string(&record.cohort.timed_preflight),
        escape_json_string(&record.cohort.postflight),
        hex_digest(&left_evidence.launch_digest()),
        hex_digest(&right_evidence.launch_digest()),
        hex_digest(&left_evidence.request_identity_digest()),
        hex_digest(&right_evidence.request_identity_digest()),
        escape_json_string(&format!("{:?}", left_evidence.route_identity())),
        escape_json_string(&format!("{:?}", right_evidence.route_identity())),
        record.cohort.left_iterations,
        record.cohort.right_iterations,
        record.summary.left_first_p05,
        record.summary.left_first_p50,
        record.summary.left_first_p95,
        record.summary.right_first_p05,
        record.summary.right_first_p50,
        record.summary.right_first_p95,
        record.summary.left_first_median_low,
        record.summary.left_first_median_high,
        record.summary.right_first_median_low,
        record.summary.right_first_median_high,
        retention_decision(record.comparison, record.summary),
        escape_json_string(&format!("{:?}", left_evidence.nodes())),
        escape_json_string(&format!("{:?}", right_evidence.nodes())),
        render_samples(&record.samples.left_first.left_us),
        render_samples(&record.samples.left_first.right_us),
        render_samples(&record.samples.left_first.ratios),
        render_samples(&record.samples.right_first.left_us),
        render_samples(&record.samples.right_first.right_us),
        render_samples(&record.samples.right_first.ratios),
    )
}

fn run_path(
    cell: Cell,
    path: Path,
    comparison: Comparison,
    left: &mut QualifiedPhysicalLaunch<'_>,
    right: &mut QualifiedPhysicalLaunch<'_>,
    runtime: &CellRuntime<'_>,
    sink: &mut JsonlSink,
) -> Result<CohortDecision, String> {
    let pair = runtime.pair;
    let label = format!("{}/{}/{}", comparison.suite, cell.name, path.name());
    let calibration_preflight = pair
        .quiet_gpu
        .require_cohort(&format!("{label}/calibration"))?;
    left.seed_f32_operands(pair.left_ctx, 0x7b31_4d29)?;
    right.seed_f32_operands(pair.right_ctx, 0x7b31_4d29)?;
    measure(left, pair.left_ctx, path, WARMUPS)?;
    measure(right, pair.right_ctx, path, WARMUPS)?;
    let left_iterations = calibrate(left, pair.left_ctx, path)?;
    let right_iterations = calibrate(right, pair.right_ctx, path)?;
    let mut samples = PairSamples::with_capacity(WINDOWS);
    let timed_preflight = pair.quiet_gpu.require_cohort(&format!("{label}/timed"))?;
    for _ in 0..WINDOWS {
        let left_us = measure(left, pair.left_ctx, path, left_iterations)?;
        let right_us = measure(right, pair.right_ctx, path, right_iterations)?;
        samples.left_first.push(left_us, right_us)?;

        let right_us = measure(right, pair.right_ctx, path, right_iterations)?;
        let left_us = measure(left, pair.left_ctx, path, left_iterations)?;
        samples.right_first.push(left_us, right_us)?;
    }
    let postflight = pair
        .quiet_gpu
        .verify_post_cohort(&format!("{label}/post"))?;
    let summary = summarize(&samples)?;
    let retained = retention_decision(comparison, summary);
    let cohort = CohortEvidence {
        calibration_preflight,
        timed_preflight,
        postflight,
        left_iterations,
        right_iterations,
    };
    sink.write(emit_evidence(
        EvidenceRecord {
            cell,
            path,
            comparison,
            runtime,
            samples: &samples,
            summary,
            cohort: &cohort,
        },
        left,
        right,
    ))?;
    Ok(CohortDecision {
        comparison: comparison.suite,
        cell: cell.name,
        path: path.name(),
        retained,
    })
}

fn run_comparison(
    device: &GpuDevice,
    quiet_gpu: &QuietGpu,
    assignment_preflight_snapshot: &str,
    comparison: Comparison,
    sink: &mut JsonlSink,
) -> Result<Vec<CohortDecision>, String> {
    let left_ctx = context(device)?;
    let right_ctx = context(device)?;
    let left_requests = CELLS.map(|cell| request(cell, comparison.left));
    let right_requests = CELLS.map(|cell| request(cell, comparison.right));
    presize_physical_qualification_suite(&left_ctx, &left_requests)?;
    presize_physical_qualification_suite(&right_ctx, &right_requests)?;
    let runtime = PairRuntime {
        left_ctx: &left_ctx,
        right_ctx: &right_ctx,
        quiet_gpu,
        assignment_preflight_snapshot,
        multiprocessors: device.multiprocessor_count(),
    };
    let mut decisions = Vec::with_capacity(CELLS.len() * 2);
    for cell in CELLS {
        let mut left = qualify_physical_launch(&left_ctx, request(cell, comparison.left))?;
        let mut right = qualify_physical_launch(&right_ctx, request(cell, comparison.right))?;
        assert_manifest(cell, comparison.left, &left)?;
        assert_manifest(cell, comparison.right, &right)?;
        let semantic_output_digest = semantic_gate(cell, &mut left, &mut right, &runtime)?;
        let correctness = dense_determinism_gate(
            cell,
            comparison,
            &mut left,
            &mut right,
            &runtime,
            semantic_output_digest,
        )?;
        let cell_runtime = CellRuntime {
            pair: &runtime,
            correctness: &correctness,
        };
        for path in [Path::Eager, Path::Graph] {
            decisions.push(run_path(
                cell,
                path,
                comparison,
                &mut left,
                &mut right,
                &cell_runtime,
                sink,
            )?);
        }
    }
    Ok(decisions)
}

fn validate_context_balance(
    decisions: &[CohortDecision],
    label: &str,
    comparisons: [&str; 2],
) -> Result<(), String> {
    if decisions.len() != CELLS.len() * 4 {
        return Err(format!(
            "{label} requires {} context-balanced sets, received {}",
            CELLS.len() * 4,
            decisions.len()
        ));
    }
    for cell in CELLS {
        for path in [Path::Eager, Path::Graph] {
            for comparison in comparisons {
                let count = decisions
                    .iter()
                    .filter(|decision| {
                        decision.comparison == comparison
                            && decision.cell == cell.name
                            && decision.path == path.name()
                    })
                    .count();
                if count != 1 {
                    return Err(format!(
                        "{label} {} {} requires one {comparison} set, received {count}",
                        cell.name,
                        path.name()
                    ));
                }
            }
        }
    }
    Ok(())
}

fn require_graph_win_and_eager_shape_crossover(decisions: &[CohortDecision]) -> Result<(), String> {
    validate_context_balance(
        decisions,
        "split-K4/direct retention",
        [SPLIT_DIRECT.suite, DIRECT_SPLIT.suite],
    )?;
    for decision in decisions {
        let accepted = if decision.path == Path::Graph.name() {
            decision.retained == Arm::SplitS4.name()
        } else if decision.cell == "m64_k1536_n384" {
            decision.retained != Arm::DirectS4.name()
        } else {
            decision.cell == "m64_k833_n384" && decision.retained == Arm::DirectS4.name()
        };
        if !accepted {
            return Err(format!(
                "unexpected split-K4/direct crossover at {}/{}/{}: {}",
                decision.comparison, decision.cell, decision.path, decision.retained
            ));
        }
    }
    Ok(())
}

fn qualification_device() -> (QuietGpu, String, GpuDevice) {
    validate_release_build(cfg!(debug_assertions))
        .expect("split-K4 performance qualification requires --release");
    let quiet_gpu = QuietGpu::for_cuda_ordinal(0).expect("resolve CUDA device 0 UUID");
    let pre_context_snapshot = quiet_gpu
        .require_pre_context("splitk4/pre-context")
        .expect("exclusive CUDA device 0 before context creation");
    let device = GpuDevice::new(0).expect("CUDA device 0");
    validate_device(&device).expect("exact SM89 or SM120 performance admission");
    (quiet_gpu, pre_context_snapshot, device)
}

#[test]
fn graph_win_and_eager_shape_crossover_require_each_orientation_once() {
    let mut decisions = Vec::new();
    for cell in CELLS {
        for path in [Path::Eager, Path::Graph] {
            for comparison in [SPLIT_DIRECT, DIRECT_SPLIT] {
                decisions.push(CohortDecision {
                    comparison: comparison.suite,
                    cell: cell.name,
                    path: path.name(),
                    retained: if path.name() == Path::Graph.name() {
                        Arm::SplitS4.name()
                    } else if cell.name == "m64_k1536_n384" {
                        "inconclusive"
                    } else {
                        Arm::DirectS4.name()
                    },
                });
            }
        }
    }
    assert!(require_graph_win_and_eager_shape_crossover(&decisions).is_ok());
    decisions[0].retained = Arm::DirectS4.name();
    assert!(require_graph_win_and_eager_shape_crossover(&decisions).is_err());
    decisions[0].retained = "inconclusive";
    decisions[0].comparison = DIRECT_SPLIT.suite;
    assert!(require_graph_win_and_eager_shape_crossover(&decisions).is_err());
}

#[test]
#[ignore = "requires an otherwise idle qualified SM89 or SM120 GPU; run with one test thread"]
fn splitk4_graph_win_and_eager_shape_crossover_on_deep_reductions() {
    let (quiet_gpu, pre_context_snapshot, device) = qualification_device();
    let mut sink = JsonlSink::create_from_env().expect("new split-K4/direct evidence file");
    let mut decisions = run_comparison(
        &device,
        &quiet_gpu,
        &pre_context_snapshot,
        SPLIT_DIRECT,
        &mut sink,
    )
    .unwrap();
    let swapped_assignment_preflight = quiet_gpu
        .require_cohort("splitk4/direct/swapped-assignment")
        .expect("quiet CUDA device before swapped direct context assignment");
    decisions.extend(
        run_comparison(
            &device,
            &quiet_gpu,
            &swapped_assignment_preflight,
            DIRECT_SPLIT,
            &mut sink,
        )
        .unwrap(),
    );
    require_graph_win_and_eager_shape_crossover(&decisions).unwrap();
    sink.finish(
        decisions.len(),
        "graph_splitk4_eager_shape_crossover",
        &quiet_gpu,
        &device,
    )
    .unwrap();
}
