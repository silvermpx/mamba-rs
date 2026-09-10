#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use cudarc::driver::{CudaGraph, sys};
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_inference::{
    InferenceFwdOperands, InferenceShape, InferenceSm120HalfTile, InferenceTile, inference_forward,
    inference_forward_with_tile,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    ArtifactIdentity, ArtifactKind, CompilerIdentity, DeviceIdentity, ModuleKind,
    TUNING_TABLE_REVISION, digest_hex,
};
use sha2::{Digest as _, Sha256};

const OUTPUT_ENV: &str = "MAMBA_RS_FIXED_HALF_M64N128_JSONL";
const SCHEMA: &str = "MambaBiFixedHalfM64N128QualificationV1";
const QUALIFICATION_SOURCE: &str = include_str!("gemm_bi_inference_half_m64n128_qualification.rs");
const RUST_DISPATCH_SOURCE: &str = include_str!("../../src/mamba_ssm/gpu/gemm_bi_inference.rs");
const CUDA_SOURCE: &str = include_str!("../../kernels/gemm_bi_inference/sm120_tma.cu");
const CANDIDATE_TILE: InferenceSm120HalfTile = InferenceSm120HalfTile::M64N128Bk64S2;
const EAGER_REPEATS: usize = 10;
const GRAPH_WARMUPS: usize = 1;
const GRAPH_REPLAYS: usize = 10;
const TIMING_WARMUPS: usize = 128;
const PILOT_ITERATIONS: usize = 16;
const WINDOWS_PER_ORDER: usize = 101;
const TARGET_WINDOW_US: f64 = 5_000.0;
const MAX_WINDOW_ITERATIONS: usize = 4_096;
const RECORDS_PER_CELL: usize = 2 + 4 + 3 + (4 * WINDOWS_PER_ORDER) + 4 + 4 + 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BiasKind {
    None,
    Synthesized,
}

impl BiasKind {
    const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Synthesized => "synthesized",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct QualCell {
    shape_name: &'static str,
    shape: InferenceShape,
    dtype: WeightDtype,
    bias: BiasKind,
}

const SHAPES: [(&str, InferenceShape); 12] = [
    (
        "m512_k1928_n2304",
        InferenceShape {
            m: 512,
            k: 1928,
            n: 2304,
        },
    ),
    (
        "m1536_k1928_n2304",
        InferenceShape {
            m: 1536,
            k: 1928,
            n: 2304,
        },
    ),
    (
        "m1536_k1928_n1928",
        InferenceShape {
            m: 1536,
            k: 1928,
            n: 1928,
        },
    ),
    (
        "m3072_k1928_n1928",
        InferenceShape {
            m: 3072,
            k: 1928,
            n: 1928,
        },
    ),
    (
        "m4096_k520_n1536",
        InferenceShape {
            m: 4096,
            k: 520,
            n: 1536,
        },
    ),
    (
        "m2048_k1928_n1536",
        InferenceShape {
            m: 2048,
            k: 1928,
            n: 1536,
        },
    ),
    (
        "m1024_k1928_n1928",
        InferenceShape {
            m: 1024,
            k: 1928,
            n: 1928,
        },
    ),
    (
        "m1024_k1928_n2304",
        InferenceShape {
            m: 1024,
            k: 1928,
            n: 2304,
        },
    ),
    (
        "m1536_k1032_n1536",
        InferenceShape {
            m: 1536,
            k: 1032,
            n: 1536,
        },
    ),
    (
        "m1536_k1928_n1536",
        InferenceShape {
            m: 1536,
            k: 1928,
            n: 1536,
        },
    ),
    (
        "m4621_k1928_n1928",
        InferenceShape {
            m: 4621,
            k: 1928,
            n: 1928,
        },
    ),
    (
        "m1536_k768_n1536",
        InferenceShape {
            m: 1536,
            k: 768,
            n: 1536,
        },
    ),
];

fn qualification_cells() -> Vec<QualCell> {
    let mut cells = Vec::with_capacity(SHAPES.len() * 4);
    for (shape_name, shape) in SHAPES {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for bias in [BiasKind::None, BiasKind::Synthesized] {
                cells.push(QualCell {
                    shape_name,
                    shape,
                    dtype,
                    bias,
                });
            }
        }
    }
    cells
}

fn dtype_name(dtype: WeightDtype) -> &'static str {
    match dtype {
        WeightDtype::Bf16 => "bf16",
        WeightDtype::F16 => "f16",
        WeightDtype::F32 => "f32",
    }
}

fn cell_id(cell: QualCell) -> String {
    format!(
        "{}_{}_{}",
        cell.shape_name,
        dtype_name(cell.dtype),
        cell.bias.name()
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Candidate,
    ProductionAuto,
    FastCublas,
}

impl Arm {
    const fn name(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::ProductionAuto => "production_auto",
            Self::FastCublas => "fast_cublas",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComparisonKind {
    ProductionAuto,
    FastCublas,
}

impl ComparisonKind {
    const fn name(self) -> &'static str {
        match self {
            Self::ProductionAuto => "candidate_vs_production_auto",
            Self::FastCublas => "candidate_vs_fast_cublas",
        }
    }

    const fn comparator(self) -> Arm {
        match self {
            Self::ProductionAuto => Arm::ProductionAuto,
            Self::FastCublas => Arm::FastCublas,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PairOrder {
    CandidateFirst,
    ComparatorFirst,
}

impl PairOrder {
    const fn name(self, kind: ComparisonKind) -> &'static str {
        match (self, kind) {
            (Self::CandidateFirst, ComparisonKind::ProductionAuto) => {
                "candidate_then_production_auto"
            }
            (Self::ComparatorFirst, ComparisonKind::ProductionAuto) => {
                "production_auto_then_candidate"
            }
            (Self::CandidateFirst, ComparisonKind::FastCublas) => "candidate_then_fast_cublas",
            (Self::ComparatorFirst, ComparisonKind::FastCublas) => "fast_cublas_then_candidate",
        }
    }
}

struct JsonlSink {
    path: PathBuf,
    writer: BufWriter<File>,
    digest: Sha256,
    records: usize,
}

impl JsonlSink {
    fn create_new(path: PathBuf) -> Result<Self, String> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| format!("create dedicated qualification sink {path:?}: {error}"))?;
        let metadata = file
            .metadata()
            .map_err(|error| format!("inspect qualification sink {path:?}: {error}"))?;
        if !metadata.file_type().is_file() {
            return Err(format!("qualification sink {path:?} is not a regular file"));
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
            .map_err(|error| format!("write qualification sink {:?}: {error}", self.path))?;
        self.digest.update(&bytes);
        self.records += 1;
        Ok(())
    }

    fn finish(mut self, expected_before_completion: usize) -> Result<(), String> {
        if self.records != expected_before_completion {
            return Err(format!(
                "incomplete qualification JSONL: got {} records before completion, expected {expected_before_completion}",
                self.records
            ));
        }
        let content_digest = format!("{:x}", self.digest.clone().finalize());
        self.write(format!(
            concat!(
                "{{\"schema\":\"{}\",\"record_type\":\"completion\",",
                "\"arm\":\"run\",\"order\":\"not_applicable\",",
                "\"path\":\"completion\",\"bias\":\"not_applicable\",",
                "\"records_before_completion\":{},\"expected_records_before_completion\":{},",
                "\"expected_records_total\":{},\"content_sha256_before_completion\":\"{}\",",
                "\"complete\":true}}"
            ),
            json_escape(SCHEMA),
            expected_before_completion,
            expected_before_completion,
            expected_before_completion + 1,
            json_escape(&content_digest),
        ))?;
        self.writer
            .flush()
            .map_err(|error| format!("flush qualification sink {:?}: {error}", self.path))?;
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("sync qualification sink {:?}: {error}", self.path))?;
        eprintln!(
            "Fixed HALF M64N128 qualification wrote {} records to {:?}",
            self.records, self.path
        );
        Ok(())
    }
}

#[derive(Clone)]
struct RunMetadata {
    run_identity: String,
    compiler: CompilerIdentity,
    artifact: ArtifactIdentity,
    artifact_set_digest: String,
    device: DeviceIdentity,
    nvrtc_target: String,
    qualification_source_sha256: String,
    rust_dispatch_source_sha256: String,
    cuda_source_sha256: String,
    executable_sha256: String,
}

impl RunMetadata {
    fn new(ctx: &GpuCtx, device: &GpuDevice) -> Result<Self, String> {
        let compiler = ctx.kernels.compiler_identity();
        let artifacts = ctx.kernels.artifact_set_identity();
        let identity = device.identity();
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let qualification_source_sha256 = verify_compiled_source(
            &manifest.join("tests/gemm_bi_inference_half_m64n128_qualification.rs"),
            QUALIFICATION_SOURCE,
        )?;
        let rust_dispatch_source_sha256 = verify_compiled_source(
            &manifest.join("src/mamba_ssm/gpu/gemm_bi_inference.rs"),
            RUST_DISPATCH_SOURCE,
        )?;
        let cuda_source_sha256 = verify_compiled_source(
            &manifest.join("kernels/gemm_bi_inference/sm120_tma.cu"),
            CUDA_SOURCE,
        )?;
        let executable_sha256 = sha256_file(
            &std::env::current_exe()
                .map_err(|error| format!("resolve qualification executable: {error}"))?,
        )?;
        let mut hasher = Sha256::new();
        hasher.update(compiler.source_digest);
        hasher.update(compiler.invocation_digest);
        hasher.update(artifacts.fixed.compile_key);
        hasher.update(artifacts.fixed.artifact_digest);
        hasher.update(artifacts.ordered_digest);
        hasher.update(identity.compute_capability.0.to_le_bytes());
        hasher.update(identity.compute_capability.1.to_le_bytes());
        hasher.update(identity.multiprocessor_count.to_le_bytes());
        hasher.update(identity.driver.api_version.to_le_bytes());
        hasher.update(identity.driver.build_digest);
        hasher.update(TUNING_TABLE_REVISION.to_le_bytes());
        hasher.update(qualification_source_sha256.as_bytes());
        hasher.update(rust_dispatch_source_sha256.as_bytes());
        hasher.update(cuda_source_sha256.as_bytes());
        hasher.update(executable_sha256.as_bytes());
        Ok(Self {
            run_identity: format!("{:x}", hasher.finalize()),
            compiler,
            artifact: artifacts.fixed,
            artifact_set_digest: digest_hex(&artifacts.ordered_digest),
            device: identity,
            nvrtc_target: device.nvrtc_target().to_owned(),
            qualification_source_sha256,
            rust_dispatch_source_sha256,
            cuda_source_sha256,
            executable_sha256,
        })
    }
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| format!("open {path:?}: {error}"))?;
    let mut hasher = Sha256::new();
    let mut chunk = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut chunk)
            .map_err(|error| format!("read {path:?}: {error}"))?;
        if count == 0 {
            break;
        }
        hasher.update(&chunk[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn verify_compiled_source(path: &Path, compiled: &str) -> Result<String, String> {
    let compiled_digest = sha256_bytes(compiled.as_bytes());
    let disk_digest = sha256_file(path)?;
    if compiled_digest != disk_digest {
        return Err(format!(
            "compiled source for {path:?} differs from the on-disk source"
        ));
    }
    Ok(compiled_digest)
}

struct CellBuffers {
    a: DtypedBuf,
    b: DtypedBuf,
    bias: DtypedBuf,
    candidate: DtypedBuf,
    production: DtypedBuf,
    vendor: DtypedBuf,
    poison_values: Vec<f32>,
    poison_bits: Vec<u32>,
}

impl CellBuffers {
    fn new(ctx: &GpuCtx, cell: QualCell) -> Result<Self, String> {
        let shape = cell.shape;
        let output_len = shape.m * shape.n;
        let a = DtypedBuf::zeros(&ctx.stream, shape.m * shape.k, cell.dtype)?;
        let b = DtypedBuf::zeros(&ctx.stream, shape.k * shape.n, cell.dtype)?;
        let bias = DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32)?;
        let candidate = DtypedBuf::zeros(&ctx.stream, output_len, cell.dtype)?;
        let production = DtypedBuf::zeros(&ctx.stream, output_len, cell.dtype)?;
        let vendor = DtypedBuf::zeros(&ctx.stream, output_len, cell.dtype)?;
        a.upload_f32(
            &ctx.stream,
            &synth_values(shape.m * shape.k, 0xa170_7f32_0000_0001),
        )?;
        b.upload_f32(
            &ctx.stream,
            &synth_values(shape.k * shape.n, 0xb170_7f32_0000_0002),
        )?;
        bias.upload_f32(&ctx.stream, &synth_values(shape.n, 0xb1a5_7f32_0000_0003))?;
        let poison_values = (0..output_len)
            .map(|index| f32::from_bits(0x7fc0_0000 | (index as u32 & 0x003f_ffff)))
            .collect::<Vec<_>>();
        candidate.upload_f32(&ctx.stream, &poison_values)?;
        let poison_bits = output_bits(ctx, &candidate, output_len)?;
        Ok(Self {
            a,
            b,
            bias,
            candidate,
            production,
            vendor,
            poison_values,
            poison_bits,
        })
    }

    fn output(&self, arm: Arm) -> &DtypedBuf {
        match arm {
            Arm::Candidate => &self.candidate,
            Arm::ProductionAuto => &self.production,
            Arm::FastCublas => &self.vendor,
        }
    }

    fn operands(&self, cell: QualCell, arm: Arm) -> InferenceFwdOperands {
        let typed_ptr = |buffer: &DtypedBuf| TypedPtr {
            ptr: buffer.cached_ptr(),
            dtype: cell.dtype,
        };
        InferenceFwdOperands {
            c: typed_ptr(self.output(arm)),
            x: typed_ptr(&self.a),
            w: typed_ptr(&self.b),
            bias_ptr: (cell.bias == BiasKind::Synthesized).then(|| self.bias.cached_ptr()),
        }
    }

    fn reset(&self, ctx: &GpuCtx, arm: Arm) -> Result<(), String> {
        self.output(arm)
            .upload_f32(&ctx.stream, &self.poison_values)
    }
}

#[derive(Clone, Debug)]
struct PhysicalSnapshot {
    symbol: String,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dynamic_shared_bytes: u32,
    registers: i32,
    local_bytes: i32,
    static_shared_bytes: i32,
    max_dynamic_shared_bytes: i32,
    preferred_shared_carveout: i32,
    occupancy_blocks_per_sm: i32,
}

#[derive(Clone, Copy, Debug)]
struct FrozenIterations {
    candidate: usize,
    production: usize,
    vendor: usize,
}

impl FrozenIterations {
    const fn for_arm(self, arm: Arm) -> usize {
        match arm {
            Arm::Candidate => self.candidate,
            Arm::ProductionAuto => self.production,
            Arm::FastCublas => self.vendor,
        }
    }
}

#[derive(Clone, Debug)]
struct PairSummary {
    kind: ComparisonKind,
    order: PairOrder,
    candidate_p50_us: f64,
    candidate_p95_us: f64,
    comparator_p50_us: f64,
    comparator_p95_us: f64,
    ratio_p50: f64,
    ratio_p95: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Aggregate {
    summaries: usize,
    candidate_first_summaries: usize,
    comparator_first_summaries: usize,
    stable_ratio_wins: usize,
    candidate_p50_us: f64,
    candidate_p95_us: f64,
    comparator_p50_us: f64,
    comparator_p95_us: f64,
}

impl Aggregate {
    fn add(&mut self, summary: &PairSummary) {
        self.summaries += 1;
        match summary.order {
            PairOrder::CandidateFirst => self.candidate_first_summaries += 1,
            PairOrder::ComparatorFirst => self.comparator_first_summaries += 1,
        }
        if summary.ratio_p50 < 1.0 && summary.ratio_p95 < 1.0 {
            self.stable_ratio_wins += 1;
        }
        self.candidate_p50_us += summary.candidate_p50_us;
        self.candidate_p95_us += summary.candidate_p95_us;
        self.comparator_p50_us += summary.comparator_p50_us;
        self.comparator_p95_us += summary.comparator_p95_us;
    }

    fn safe_win(self) -> bool {
        self.summaries > 0
            && self.candidate_first_summaries == self.comparator_first_summaries
            && self.candidate_first_summaries + self.comparator_first_summaries == self.summaries
            && self.stable_ratio_wins == self.summaries
    }

    fn p50_ratio(self) -> f64 {
        self.candidate_p50_us / self.comparator_p50_us
    }

    fn p95_ratio(self) -> f64 {
        self.candidate_p95_us / self.comparator_p95_us
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CellRetentionDecision {
    production: Aggregate,
    vendor: Aggregate,
}

impl CellRetentionDecision {
    fn from_summaries(summaries: &[PairSummary]) -> Self {
        let mut decision = Self::default();
        for summary in summaries {
            match summary.kind {
                ComparisonKind::ProductionAuto => decision.production.add(summary),
                ComparisonKind::FastCublas => decision.vendor.add(summary),
            }
        }
        decision
    }

    fn production_safe_win(self) -> bool {
        self.production.summaries == 2 && self.production.safe_win()
    }

    fn vendor_observation(self) -> &'static str {
        if self.vendor.summaries == 2 && self.vendor.safe_win() {
            "candidate_stable_win"
        } else {
            "no_candidate_stable_win"
        }
    }

    fn retained_arm(self) -> Arm {
        if self.production_safe_win() {
            Arm::Candidate
        } else {
            Arm::ProductionAuto
        }
    }
}

#[derive(Default)]
struct CumulativeResults {
    all_production: Aggregate,
    all_vendor: Aggregate,
    candidate_retained_cells: usize,
    production_retained_cells: usize,
}

impl CumulativeResults {
    fn add(&mut self, summary: &PairSummary) {
        let all = match summary.kind {
            ComparisonKind::ProductionAuto => &mut self.all_production,
            ComparisonKind::FastCublas => &mut self.all_vendor,
        };
        all.add(summary);
    }

    fn add_cell_decision(&mut self, decision: CellRetentionDecision) {
        if decision.production_safe_win() {
            self.candidate_retained_cells += 1;
        } else {
            self.production_retained_cells += 1;
        }
    }

    fn disposition(&self) -> &'static str {
        match (
            self.candidate_retained_cells,
            self.production_retained_cells,
        ) {
            (0, 0) => "no_cell_decisions",
            (0, _) => "retain_production_auto_all_cells",
            (_, 0) => "retain_candidate_all_cells",
            _ => "mixed_local_retention",
        }
    }
}

fn synth_values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
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

fn configure_arm(ctx: &GpuCtx, arm: Arm) -> Result<(), String> {
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_bi_tensor_cores(false);
    match arm {
        Arm::Candidate | Arm::ProductionAuto => {
            ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        }
        Arm::FastCublas => {
            ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
        }
    }
    Ok(())
}

fn launch_arm(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    arm: Arm,
) -> Result<(), String> {
    match arm {
        Arm::Candidate => inference_forward_with_tile(
            ctx,
            operands,
            shape,
            InferenceTile::Sm120Half(CANDIDATE_TILE),
        ),
        Arm::ProductionAuto => {
            let selected = inference_forward(
                ctx,
                operands.c,
                operands.x,
                operands.w,
                operands.bias_ptr,
                (shape.m, shape.k, shape.n),
            )?;
            if !matches!(selected, InferenceTile::Sm120Half(_)) {
                return Err(format!(
                    "production Fixed AUTO selected {selected:?}, expected an SM120 half tile"
                ));
            }
            Ok(())
        }
        Arm::FastCublas => gpu_gemm_typed_forward_raw(
            ctx,
            operands.c,
            operands.x,
            operands.w,
            operands.bias_ptr,
            (shape.m, shape.k, shape.n),
        ),
    }
}

fn output_bits(ctx: &GpuCtx, output: &DtypedBuf, len: usize) -> Result<Vec<u32>, String> {
    ctx.stream
        .synchronize()
        .map_err(|error| format!("synchronize output download: {error:?}"))?;
    let mut values = vec![0.0f32; len];
    output.download_f32(&ctx.stream, &mut values)?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("synchronize downloaded output: {error:?}"))?;
    Ok(values.into_iter().map(f32::to_bits).collect())
}

fn bits_digest(bits: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for word in bits {
        hasher.update(word.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn eager_bits(
    ctx: &GpuCtx,
    cell: QualCell,
    buffers: &CellBuffers,
    arm: Arm,
    reference: Option<&[u32]>,
) -> Result<Vec<u32>, String> {
    configure_arm(ctx, arm)?;
    let operands = buffers.operands(cell, arm);
    buffers.reset(ctx, arm)?;
    launch_arm(ctx, operands, cell.shape, arm)?;
    let first = output_bits(ctx, buffers.output(arm), cell.shape.m * cell.shape.n)?;
    if first
        .iter()
        .zip(&buffers.poison_bits)
        .any(|(actual, poison)| actual == poison)
    {
        return Err(format!(
            "{} {} eager output retained a poison element",
            cell_id(cell),
            arm.name()
        ));
    }
    if reference.is_some_and(|expected| first != expected) {
        return Err(format!(
            "{} {} eager bits differ",
            cell_id(cell),
            arm.name()
        ));
    }
    for repeat in 0..EAGER_REPEATS {
        buffers.reset(ctx, arm)?;
        launch_arm(ctx, operands, cell.shape, arm)?;
        let actual = output_bits(ctx, buffers.output(arm), cell.shape.m * cell.shape.n)?;
        if actual != first {
            return Err(format!(
                "{} {} eager repeat {repeat} changed bits",
                cell_id(cell),
                arm.name()
            ));
        }
    }
    Ok(first)
}

fn capture_arm_graph(
    ctx: &GpuCtx,
    cell: QualCell,
    buffers: &CellBuffers,
    arm: Arm,
) -> Result<CudaGraph, String> {
    configure_arm(ctx, arm)?;
    let operands = buffers.operands(cell, arm);
    unsafe { capture_into_graph(&ctx.stream, || launch_arm(ctx, operands, cell.shape, arm)) }
}

fn graph_bits(
    ctx: &GpuCtx,
    cell: QualCell,
    buffers: &CellBuffers,
    arm: Arm,
    graph: &CudaGraph,
    reference: &[u32],
) -> Result<(), String> {
    for replay in 0..(GRAPH_WARMUPS + GRAPH_REPLAYS) {
        buffers.reset(ctx, arm)?;
        graph
            .launch()
            .map_err(|error| format!("{} {} graph replay: {error:?}", cell_id(cell), arm.name()))?;
        let actual = output_bits(ctx, buffers.output(arm), cell.shape.m * cell.shape.n)?;
        if actual != reference {
            return Err(format!(
                "{} {} graph replay {replay} changed bits",
                cell_id(cell),
                arm.name()
            ));
        }
    }
    Ok(())
}

fn cuda_ok(result: sys::CUresult, operation: &str) -> Result<(), String> {
    if result == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation} failed: {result:?}"))
    }
}

fn function_attribute(
    function: sys::CUfunction,
    attribute: sys::CUfunction_attribute,
    label: &str,
) -> Result<i32, String> {
    let mut value = 0;
    cuda_ok(
        unsafe { sys::cuFuncGetAttribute(&mut value, attribute, function) },
        label,
    )?;
    Ok(value)
}

fn physical_snapshot(graph: &CudaGraph) -> Result<PhysicalSnapshot, String> {
    let mut node_count = 0usize;
    cuda_ok(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut node_count) },
        "query graph node count",
    )?;
    if node_count != 1 {
        return Err(format!(
            "qualification graph contains {node_count} nodes, expected one physical kernel"
        ));
    }
    let mut nodes = vec![std::ptr::null_mut(); node_count];
    cuda_ok(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut node_count) },
        "query graph nodes",
    )?;
    let mut params = unsafe { std::mem::zeroed() };
    cuda_ok(
        unsafe { sys::cuGraphKernelNodeGetParams_v2(nodes[0], &mut params) },
        "query graph kernel parameters",
    )?;
    let mut name = std::ptr::null();
    cuda_ok(
        unsafe { sys::cuFuncGetName(&mut name, params.func) },
        "query physical function name",
    )?;
    if name.is_null() {
        return Err("physical function name is null".into());
    }
    let symbol = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|error| format!("physical function name is not UTF-8: {error}"))?
        .to_owned();
    let block_threads = params
        .blockDimX
        .checked_mul(params.blockDimY)
        .and_then(|value| value.checked_mul(params.blockDimZ))
        .ok_or("physical block size overflow")?;
    let mut occupancy = 0;
    cuda_ok(
        unsafe {
            sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut occupancy,
                params.func,
                i32::try_from(block_threads).map_err(|_| "physical block size exceeds i32")?,
                params.sharedMemBytes as usize,
            )
        },
        "query physical occupancy",
    )?;
    Ok(PhysicalSnapshot {
        symbol,
        grid: (params.gridDimX, params.gridDimY, params.gridDimZ),
        block: (params.blockDimX, params.blockDimY, params.blockDimZ),
        dynamic_shared_bytes: params.sharedMemBytes,
        registers: function_attribute(
            params.func,
            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
            "query physical registers",
        )?,
        local_bytes: function_attribute(
            params.func,
            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            "query physical local bytes",
        )?,
        static_shared_bytes: function_attribute(
            params.func,
            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            "query physical static shared bytes",
        )?,
        max_dynamic_shared_bytes: function_attribute(
            params.func,
            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            "query physical maximum dynamic shared bytes",
        )?,
        preferred_shared_carveout: function_attribute(
            params.func,
            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            "query physical preferred shared carveout",
        )?,
        occupancy_blocks_per_sm: occupancy,
    })
}

fn half_symbol(tile: InferenceSm120HalfTile, dtype: WeightDtype) -> &'static str {
    match (tile, dtype) {
        (InferenceSm120HalfTile::M64N64Bk64S2, WeightDtype::Bf16) => {
            "gemm_bi_nn_sm120_tma_64x64_bk64_s2_bf16"
        }
        (InferenceSm120HalfTile::M64N64Bk64S2, WeightDtype::F16) => {
            "gemm_bi_nn_sm120_tma_64x64_bk64_s2_f16"
        }
        (InferenceSm120HalfTile::M64N128Bk64S2, WeightDtype::Bf16) => {
            "gemm_bi_nn_sm120_tma_64x128_bk64_s2_bf16"
        }
        (InferenceSm120HalfTile::M64N128Bk64S2, WeightDtype::F16) => {
            "gemm_bi_nn_sm120_tma_64x128_bk64_s2_f16"
        }
        (InferenceSm120HalfTile::M128N64Bk32S3, WeightDtype::Bf16) => {
            "gemm_bi_nn_sm120_tma_128x64_bk32_s3_bf16"
        }
        (InferenceSm120HalfTile::M128N64Bk32S3, WeightDtype::F16) => {
            "gemm_bi_nn_sm120_tma_128x64_bk32_s3_f16"
        }
        (InferenceSm120HalfTile::M128N128Bk32S2, WeightDtype::Bf16) => {
            "gemm_bi_nn_sm120_tma_128x128_bk32_s2_bf16"
        }
        (InferenceSm120HalfTile::M128N128Bk32S2, WeightDtype::F16) => {
            "gemm_bi_nn_sm120_tma_128x128_bk32_s2_f16"
        }
        (InferenceSm120HalfTile::M128N128Bk32S3, WeightDtype::Bf16) => {
            "gemm_bi_nn_sm120_tma_128x128_bk32_s3_bf16"
        }
        (InferenceSm120HalfTile::M128N128Bk32S3, WeightDtype::F16) => {
            "gemm_bi_nn_sm120_tma_128x128_bk32_s3_f16"
        }
        (_, WeightDtype::F32) => panic!("SM120 half qualification cannot use F32"),
    }
}

fn half_contract(tile: InferenceSm120HalfTile) -> (usize, usize, u32, u32) {
    match tile {
        InferenceSm120HalfTile::M64N64Bk64S2 => (64, 64, 128, 32_896),
        InferenceSm120HalfTile::M64N128Bk64S2 => (64, 128, 256, 49_280),
        InferenceSm120HalfTile::M128N64Bk32S3 => (128, 64, 256, 36_992),
        InferenceSm120HalfTile::M128N128Bk32S2 => (128, 128, 256, 32_896),
        InferenceSm120HalfTile::M128N128Bk32S3 => (128, 128, 256, 49_280),
    }
}

fn production_tile(
    snapshot: &PhysicalSnapshot,
    dtype: WeightDtype,
) -> Option<InferenceSm120HalfTile> {
    [
        InferenceSm120HalfTile::M64N64Bk64S2,
        InferenceSm120HalfTile::M64N128Bk64S2,
        InferenceSm120HalfTile::M128N64Bk32S3,
        InferenceSm120HalfTile::M128N128Bk32S2,
    ]
    .into_iter()
    .find(|tile| snapshot.symbol == half_symbol(*tile, dtype))
}

fn validate_physical(
    cell: QualCell,
    _compiler: CompilerIdentity,
    arm: Arm,
    snapshot: &PhysicalSnapshot,
) -> Result<(), String> {
    let tile = match arm {
        Arm::Candidate => CANDIDATE_TILE,
        Arm::ProductionAuto => production_tile(snapshot, cell.dtype).ok_or_else(|| {
            format!(
                "{} production AUTO launched an unknown or unqualified symbol {}",
                cell_id(cell),
                snapshot.symbol
            )
        })?,
        Arm::FastCublas => return Err("vendor arm has no Fixed physical route".into()),
    };
    let (bm, bn, block_threads, dynamic_shared) = half_contract(tile);
    let symbol = half_symbol(tile, cell.dtype);
    let grid = cell.shape.m.div_ceil(bm) * cell.shape.n.div_ceil(bn);
    let observed_threads = snapshot.block.0 * snapshot.block.1 * snapshot.block.2;
    if snapshot.symbol != symbol
        || observed_threads != block_threads
        || snapshot.dynamic_shared_bytes != dynamic_shared
        || snapshot.grid != (grid as u32, 1, 1)
    {
        return Err(format!(
            "{} {} physical route mismatch: {snapshot:?}",
            cell_id(cell),
            arm.name()
        ));
    }
    // The driver does not expose spill bytes separately. Zero local memory is the
    // runtime spill proxy here; offline ptxas evidence still owns stack and spill detail.
    if snapshot.registers <= 0
        || snapshot.registers > 128
        || snapshot.local_bytes != 0
        || snapshot.static_shared_bytes != 0
        || snapshot.preferred_shared_carveout != -1
        || i64::from(snapshot.max_dynamic_shared_bytes) < i64::from(snapshot.dynamic_shared_bytes)
        || snapshot.occupancy_blocks_per_sm < 2
    {
        return Err(format!(
            "{} {} failed the CUDA resource safety gate: {snapshot:?}",
            cell_id(cell),
            arm.name()
        ));
    }
    Ok(())
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                use std::fmt::Write as _;
                write!(escaped, "\\u{:04x}", character as u32)
                    .expect("writing to a String cannot fail");
            }
            character => escaped.push(character),
        }
    }
    escaped
}

fn cell_json(metadata: &RunMetadata, cell: QualCell) -> String {
    format!(
        concat!(
            "\"schema\":\"{}\",\"run_identity_sha256\":\"{}\",",
            "\"cell\":\"{}\",\"shape_name\":\"{}\",",
            "\"shape\":{{\"m\":{},\"k\":{},\"n\":{}}},",
            "\"dtype\":\"{}\",\"bias\":\"{}\""
        ),
        json_escape(SCHEMA),
        json_escape(&metadata.run_identity),
        json_escape(&cell_id(cell)),
        json_escape(cell.shape_name),
        cell.shape.m,
        cell.shape.k,
        cell.shape.n,
        json_escape(dtype_name(cell.dtype)),
        json_escape(cell.bias.name()),
    )
}

fn emit_physical(
    sink: &mut JsonlSink,
    metadata: &RunMetadata,
    cell: QualCell,
    arm: Arm,
    snapshot: &PhysicalSnapshot,
) -> Result<(), String> {
    sink.write(format!(
        concat!(
            "{{{},\"record_type\":\"physical_resource\",\"arm\":\"{}\",",
            "\"order\":\"not_applicable\",\"path\":\"graph\",\"symbol\":\"{}\",",
            "\"grid\":[{},{},{}],\"block\":[{},{},{}],",
            "\"dynamic_shared_bytes\":{},\"registers\":{},\"local_bytes\":{},",
            "\"static_shared_bytes\":{},\"max_dynamic_shared_bytes\":{},",
            "\"preferred_shared_carveout\":{},\"occupancy_blocks_per_sm\":{},",
            "\"compiler_invocation_sha256\":\"{}\",\"passed\":true}}"
        ),
        cell_json(metadata, cell),
        json_escape(arm.name()),
        json_escape(&snapshot.symbol),
        snapshot.grid.0,
        snapshot.grid.1,
        snapshot.grid.2,
        snapshot.block.0,
        snapshot.block.1,
        snapshot.block.2,
        snapshot.dynamic_shared_bytes,
        snapshot.registers,
        snapshot.local_bytes,
        snapshot.static_shared_bytes,
        snapshot.max_dynamic_shared_bytes,
        snapshot.preferred_shared_carveout,
        snapshot.occupancy_blocks_per_sm,
        json_escape(&digest_hex(&metadata.compiler.invocation_digest)),
    ))
}

fn emit_bit_gate(
    sink: &mut JsonlSink,
    metadata: &RunMetadata,
    cell: QualCell,
    arm: Arm,
    path: &str,
    digest: &str,
    production_reference_digest: &str,
) -> Result<(), String> {
    let (eager_repeats, graph_warmups, graph_replays) = if path == "eager" {
        (EAGER_REPEATS, 0, 0)
    } else {
        (0, GRAPH_WARMUPS, GRAPH_REPLAYS)
    };
    sink.write(format!(
        concat!(
            "{{{},\"record_type\":\"bit_gate\",\"arm\":\"{}\",\"path\":\"{}\",",
            "\"order\":\"not_applicable\",",
            "\"output_sha256\":\"{}\",\"reference_arm\":\"production_auto\",",
            "\"production_reference_sha256\":\"{}\",",
            "\"eager_repeats\":{},\"graph_warmups\":{},\"graph_replays\":{},",
            "\"poison_reset_each_launch\":true,\"exact_bits_equal\":true,\"passed\":true}}"
        ),
        cell_json(metadata, cell),
        json_escape(arm.name()),
        json_escape(path),
        json_escape(digest),
        json_escape(production_reference_digest),
        eager_repeats,
        graph_warmups,
        graph_replays,
    ))
}

fn run_correctness(
    sink: &mut JsonlSink,
    metadata: &RunMetadata,
    ctx: &GpuCtx,
    cell: QualCell,
    buffers: &CellBuffers,
) -> Result<(), String> {
    let production_bits = eager_bits(ctx, cell, buffers, Arm::ProductionAuto, None)?;
    let production_digest = bits_digest(&production_bits);
    let candidate_bits = eager_bits(ctx, cell, buffers, Arm::Candidate, Some(&production_bits))?;
    let candidate_digest = bits_digest(&candidate_bits);
    emit_bit_gate(
        sink,
        metadata,
        cell,
        Arm::ProductionAuto,
        "eager",
        &production_digest,
        &production_digest,
    )?;
    emit_bit_gate(
        sink,
        metadata,
        cell,
        Arm::Candidate,
        "eager",
        &candidate_digest,
        &production_digest,
    )?;

    let production_graph = capture_arm_graph(ctx, cell, buffers, Arm::ProductionAuto)?;
    let candidate_graph = capture_arm_graph(ctx, cell, buffers, Arm::Candidate)?;
    let production_physical = physical_snapshot(&production_graph)?;
    let candidate_physical = physical_snapshot(&candidate_graph)?;
    validate_physical(
        cell,
        metadata.compiler,
        Arm::ProductionAuto,
        &production_physical,
    )?;
    validate_physical(cell, metadata.compiler, Arm::Candidate, &candidate_physical)?;
    emit_physical(
        sink,
        metadata,
        cell,
        Arm::ProductionAuto,
        &production_physical,
    )?;
    emit_physical(sink, metadata, cell, Arm::Candidate, &candidate_physical)?;

    graph_bits(
        ctx,
        cell,
        buffers,
        Arm::ProductionAuto,
        &production_graph,
        &production_bits,
    )?;
    graph_bits(
        ctx,
        cell,
        buffers,
        Arm::Candidate,
        &candidate_graph,
        &production_bits,
    )?;
    emit_bit_gate(
        sink,
        metadata,
        cell,
        Arm::ProductionAuto,
        "graph",
        &production_digest,
        &production_digest,
    )?;
    emit_bit_gate(
        sink,
        metadata,
        cell,
        Arm::Candidate,
        "graph",
        &candidate_digest,
        &production_digest,
    )
}

fn measure_window(
    ctx: &GpuCtx,
    cell: QualCell,
    buffers: &CellBuffers,
    arm: Arm,
    iterations: usize,
) -> Result<f64, String> {
    if iterations == 0 {
        return Err("timing window iterations must be positive".into());
    }
    configure_arm(ctx, arm)?;
    let operands = buffers.operands(cell, arm);
    let start = ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("record {} timing start: {error:?}", arm.name()))?;
    for _ in 0..iterations {
        launch_arm(ctx, operands, cell.shape, arm)?;
    }
    let end = ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("record {} timing end: {error:?}", arm.name()))?;
    let per_call_us = f64::from(
        start
            .elapsed_ms(&end)
            .map_err(|error| format!("measure {} timing window: {error:?}", arm.name()))?,
    ) * 1_000.0
        / iterations as f64;
    if !per_call_us.is_finite() || per_call_us <= 0.0 {
        return Err(format!(
            "{} produced invalid timing sample {per_call_us}",
            arm.name()
        ));
    }
    Ok(per_call_us)
}

fn warm_arm(ctx: &GpuCtx, cell: QualCell, buffers: &CellBuffers, arm: Arm) -> Result<(), String> {
    configure_arm(ctx, arm)?;
    let operands = buffers.operands(cell, arm);
    for _ in 0..TIMING_WARMUPS {
        launch_arm(ctx, operands, cell.shape, arm)?;
    }
    ctx.stream
        .synchronize()
        .map_err(|error| format!("synchronize {} warmups: {error:?}", arm.name()))
}

fn calibrate_arm(
    ctx: &GpuCtx,
    cell: QualCell,
    buffers: &CellBuffers,
    arm: Arm,
) -> Result<usize, String> {
    let pilot_us = measure_window(ctx, cell, buffers, arm, PILOT_ITERATIONS)?;
    Ok(((TARGET_WINDOW_US / pilot_us).ceil() as usize).clamp(1, MAX_WINDOW_ITERATIONS))
}

fn nearest_rank(values: &[f64], fraction: f64) -> Result<f64, String> {
    if values.len() != WINDOWS_PER_ORDER
        || !(0.0..=1.0).contains(&fraction)
        || fraction == 0.0
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("nearest-rank percentile requires 101 positive finite samples".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Ok(sorted[index])
}

fn parse_telemetry(line: &str) -> Result<(u32, u32, u32, String), String> {
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
    if sm_clock_mhz == 0 || temperature_c == 0 {
        return Err(format!(
            "GPU telemetry preflight returned invalid row {line:?}"
        ));
    }
    let snapshot = format!(
        "gpu_util={gpu_util}% memory_util={memory_util}% used={used_mib}MiB sm_clock={sm_clock_mhz}MHz temperature={temperature_c}C pstate={}",
        fields[5]
    );
    Ok((gpu_util, memory_util, used_mib, snapshot))
}

const fn telemetry_is_quiet(gpu_util: u32, memory_util: u32) -> bool {
    gpu_util <= 1 && memory_util <= 1
}

fn competing_compute_pids() -> Result<Vec<u32>, String> {
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
    Ok(String::from_utf8_lossy(&applications.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .filter(|pid| *pid != own_pid)
        .collect())
}

fn require_quiet_gpu(label: &str) -> Result<String, String> {
    let mut last_snapshot = String::new();
    let mut last_competing_pids = Vec::new();
    for _ in 0..50 {
        let competing_before = competing_compute_pids()?;
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
        let (gpu_util, memory_util, _used_mib, snapshot) = parse_telemetry(&line)?;
        last_snapshot = snapshot;
        let competing_after = competing_compute_pids()?;
        last_competing_pids = if competing_after.is_empty() {
            competing_before.clone()
        } else {
            competing_after
        };
        if competing_before.is_empty()
            && last_competing_pids.is_empty()
            && telemetry_is_quiet(gpu_util, memory_util)
        {
            eprintln!("HALF M64N128 route-set preflight {label}: {last_snapshot}");
            return Ok(last_snapshot);
        }
        thread::sleep(Duration::from_millis(100));
    }
    Err(format!(
        "{label} did not reach an exclusive <=1% compute and memory utilization state; competing PIDs: {last_competing_pids:?}; last snapshot: {last_snapshot}"
    ))
}

fn run_paired(
    sink: &mut JsonlSink,
    metadata: &RunMetadata,
    ctx: &GpuCtx,
    cell: QualCell,
    buffers: &CellBuffers,
    iterations: FrozenIterations,
    pair: (ComparisonKind, PairOrder),
) -> Result<PairSummary, String> {
    let (kind, order) = pair;
    let comparator = kind.comparator();
    let order_name = order.name(kind);
    let preflight = require_quiet_gpu(order_name)?;
    sink.write(format!(
        concat!(
            "{{{},\"record_type\":\"cohort_preflight\",\"comparison\":\"{}\",",
            "\"arm\":\"candidate\",\"comparator_arm\":\"{}\",",
            "\"path\":\"preflight\",\"order\":\"{}\",",
            "\"foreign_compute_processes\":0,\"max_gpu_utilization_pct\":1,",
            "\"max_memory_utilization_pct\":1,\"telemetry\":\"{}\",\"passed\":true}}"
        ),
        cell_json(metadata, cell),
        json_escape(kind.name()),
        json_escape(comparator.name()),
        json_escape(order_name),
        json_escape(&preflight),
    ))?;
    let candidate_iterations = iterations.for_arm(Arm::Candidate);
    let comparator_iterations = iterations.for_arm(comparator);
    let mut candidate_samples = Vec::with_capacity(WINDOWS_PER_ORDER);
    let mut comparator_samples = Vec::with_capacity(WINDOWS_PER_ORDER);
    let mut ratios = Vec::with_capacity(WINDOWS_PER_ORDER);
    for window in 0..WINDOWS_PER_ORDER {
        let (candidate_us, comparator_us) = match order {
            PairOrder::CandidateFirst => (
                measure_window(ctx, cell, buffers, Arm::Candidate, candidate_iterations)?,
                measure_window(ctx, cell, buffers, comparator, comparator_iterations)?,
            ),
            PairOrder::ComparatorFirst => {
                let comparator_us =
                    measure_window(ctx, cell, buffers, comparator, comparator_iterations)?;
                let candidate_us =
                    measure_window(ctx, cell, buffers, Arm::Candidate, candidate_iterations)?;
                (candidate_us, comparator_us)
            }
        };
        let ratio = candidate_us / comparator_us;
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err(format!(
                "{} window {window} produced ratio {ratio}",
                cell_id(cell)
            ));
        }
        candidate_samples.push(candidate_us);
        comparator_samples.push(comparator_us);
        ratios.push(ratio);
        sink.write(format!(
            concat!(
                "{{{},\"record_type\":\"paired_window\",\"comparison\":\"{}\",",
                "\"arm\":\"candidate\",\"comparator_arm\":\"{}\",\"path\":\"eager\",",
                "\"order\":\"{}\",\"window\":{},\"candidate_iterations\":{},",
                "\"comparator_iterations\":{},\"candidate_us\":{:.9},",
                "\"comparator_us\":{:.9},\"candidate_over_comparator\":{:.9}}}"
            ),
            cell_json(metadata, cell),
            json_escape(kind.name()),
            json_escape(comparator.name()),
            json_escape(order.name(kind)),
            window,
            candidate_iterations,
            comparator_iterations,
            candidate_us,
            comparator_us,
            ratio,
        ))?;
    }
    let summary = PairSummary {
        kind,
        order,
        candidate_p50_us: nearest_rank(&candidate_samples, 0.50)?,
        candidate_p95_us: nearest_rank(&candidate_samples, 0.95)?,
        comparator_p50_us: nearest_rank(&comparator_samples, 0.50)?,
        comparator_p95_us: nearest_rank(&comparator_samples, 0.95)?,
        ratio_p50: nearest_rank(&ratios, 0.50)?,
        ratio_p95: nearest_rank(&ratios, 0.95)?,
    };
    sink.write(format!(
        concat!(
            "{{{},\"record_type\":\"paired_summary\",\"comparison\":\"{}\",",
            "\"arm\":\"candidate\",\"comparator_arm\":\"{}\",\"path\":\"eager\",",
            "\"order\":\"{}\",\"windows\":{},\"candidate_iterations\":{},",
            "\"comparator_iterations\":{},\"candidate_p50_us\":{:.9},",
            "\"candidate_p95_us\":{:.9},\"comparator_p50_us\":{:.9},",
            "\"comparator_p95_us\":{:.9},\"ratio_p50\":{:.9},\"ratio_p95\":{:.9},",
            "\"performance_asserted\":false}}"
        ),
        cell_json(metadata, cell),
        json_escape(kind.name()),
        json_escape(comparator.name()),
        json_escape(order.name(kind)),
        WINDOWS_PER_ORDER,
        candidate_iterations,
        comparator_iterations,
        summary.candidate_p50_us,
        summary.candidate_p95_us,
        summary.comparator_p50_us,
        summary.comparator_p95_us,
        summary.ratio_p50,
        summary.ratio_p95,
    ))?;
    Ok(summary)
}

fn run_timing(
    sink: &mut JsonlSink,
    metadata: &RunMetadata,
    ctx: &GpuCtx,
    cell: QualCell,
    buffers: &CellBuffers,
) -> Result<Vec<PairSummary>, String> {
    for arm in [Arm::Candidate, Arm::ProductionAuto, Arm::FastCublas] {
        warm_arm(ctx, cell, buffers, arm)?;
    }
    let iterations = FrozenIterations {
        candidate: calibrate_arm(ctx, cell, buffers, Arm::Candidate)?,
        production: calibrate_arm(ctx, cell, buffers, Arm::ProductionAuto)?,
        vendor: calibrate_arm(ctx, cell, buffers, Arm::FastCublas)?,
    };
    for arm in [Arm::Candidate, Arm::ProductionAuto, Arm::FastCublas] {
        sink.write(format!(
            concat!(
                "{{{},\"record_type\":\"calibration\",\"arm\":\"{}\",",
                "\"order\":\"not_applicable\",\"path\":\"eager\",",
                "\"warmups\":{},\"pilot_iterations\":{},",
                "\"target_window_us\":{},\"frozen_iterations\":{}}}"
            ),
            cell_json(metadata, cell),
            json_escape(arm.name()),
            TIMING_WARMUPS,
            PILOT_ITERATIONS,
            TARGET_WINDOW_US,
            iterations.for_arm(arm),
        ))?;
    }
    let mut summaries = Vec::with_capacity(4);
    for kind in [ComparisonKind::ProductionAuto, ComparisonKind::FastCublas] {
        for order in [PairOrder::CandidateFirst, PairOrder::ComparatorFirst] {
            summaries.push(run_paired(
                sink,
                metadata,
                ctx,
                cell,
                buffers,
                iterations,
                (kind, order),
            )?);
        }
    }
    Ok(summaries)
}

fn pair_summary(
    summaries: &[PairSummary],
    kind: ComparisonKind,
    order: PairOrder,
) -> Result<&PairSummary, String> {
    summaries
        .iter()
        .find(|summary| summary.kind == kind && summary.order == order)
        .ok_or_else(|| format!("missing {} {} summary", kind.name(), order.name(kind)))
}

fn emit_cell_retention(
    sink: &mut JsonlSink,
    metadata: &RunMetadata,
    cell: QualCell,
    summaries: &[PairSummary],
    decision: CellRetentionDecision,
) -> Result<(), String> {
    let production_first = pair_summary(
        summaries,
        ComparisonKind::ProductionAuto,
        PairOrder::CandidateFirst,
    )?;
    let production_second = pair_summary(
        summaries,
        ComparisonKind::ProductionAuto,
        PairOrder::ComparatorFirst,
    )?;
    let vendor_first = pair_summary(
        summaries,
        ComparisonKind::FastCublas,
        PairOrder::CandidateFirst,
    )?;
    let vendor_second = pair_summary(
        summaries,
        ComparisonKind::FastCublas,
        PairOrder::ComparatorFirst,
    )?;
    sink.write(format!(
        concat!(
            "{{{},\"record_type\":\"cell_retention\",\"arm\":\"{}\",",
            "\"order\":\"both_orders\",\"path\":\"eager\",",
            "\"production_candidate_first_p50_ratio\":{:.9},",
            "\"production_candidate_first_p95_ratio\":{:.9},",
            "\"production_comparator_first_p50_ratio\":{:.9},",
            "\"production_comparator_first_p95_ratio\":{:.9},",
            "\"production_safe_win\":{},\"retained_arm\":\"{}\",",
            "\"vendor_candidate_first_p50_ratio\":{:.9},",
            "\"vendor_candidate_first_p95_ratio\":{:.9},",
            "\"vendor_comparator_first_p50_ratio\":{:.9},",
            "\"vendor_comparator_first_p95_ratio\":{:.9},",
            "\"vendor_observation\":\"{}\",\"vendor_retention_effect\":false,",
            "\"policy\":\"per_cell_production_both_orders_strict_p50_p95\"}}"
        ),
        cell_json(metadata, cell),
        json_escape(decision.retained_arm().name()),
        production_first.ratio_p50,
        production_first.ratio_p95,
        production_second.ratio_p50,
        production_second.ratio_p95,
        decision.production_safe_win(),
        json_escape(decision.retained_arm().name()),
        vendor_first.ratio_p50,
        vendor_first.ratio_p95,
        vendor_second.ratio_p50,
        vendor_second.ratio_p95,
        json_escape(decision.vendor_observation()),
    ))
}

fn artifact_kind_name(kind: ArtifactKind) -> &'static str {
    match kind {
        ArtifactKind::Ptx => "ptx",
        ArtifactKind::Cubin => "cubin",
    }
}

fn emit_run_preflight(sink: &mut JsonlSink, metadata: &RunMetadata) -> Result<(), String> {
    let compiler = metadata.compiler;
    let artifact = metadata.artifact;
    let device = metadata.device;
    sink.write(format!(
        concat!(
            "{{\"schema\":\"{}\",\"record_type\":\"run_preflight\",",
            "\"arm\":\"run\",\"order\":\"not_applicable\",",
            "\"path\":\"preflight\",\"bias\":\"not_applicable\",",
            "\"run_identity_sha256\":\"{}\",\"package_version\":\"{}\",",
            "\"release_build\":true,\"tuning_table_revision\":{},\"cells\":{},",
            "\"device\":{{\"cc\":\"{}.{}\",\"sm_count\":{},\"nvrtc_target\":\"{}\",",
            "\"driver_api_version\":{},\"driver_build_sources\":{},",
            "\"driver_build_sha256\":\"{}\"}},",
            "\"fixed_compiler\":{{\"nvrtc_version\":\"{}.{}\",\"target\":\"{}\",",
            "\"source_sha256\":\"{}\",\"invocation_sha256\":\"{}\",",
            "\"header_manifest_sha256\":\"{}\",\"nvrtc_library_sha256\":\"{}\",",
            "\"nvrtc_library_known\":{},\"output_kind\":\"{}\",",
            "\"composer_revision\":{},\"compiler_revision\":{},",
            "\"numeric_abi_revision\":{},\"schedule_revision\":{}}},",
            "\"fixed_artifact\":{{\"kind\":\"{}\",\"compile_key_sha256\":\"{}\",",
            "\"artifact_sha256\":\"{}\",\"artifact_set_sha256\":\"{}\"}},",
            "\"qualification_source_sha256\":\"{}\",",
            "\"rust_dispatch_source_sha256\":\"{}\",",
            "\"cuda_source_sha256\":\"{}\",\"executable_sha256\":\"{}\",",
            "\"candidate_symbols\":[\"{}\",\"{}\"],",
            "\"candidate_tile\":\"M64N128Bk64S2\",",
            "\"production_route\":\"Fixed_AUTO\",",
            "\"vendor_route\":\"fast_cublas_half\",\"timing_warmups\":{},",
            "\"windows_per_order\":{},\"orders_per_comparison\":2,",
            "\"atomics\":false,\"split_k\":false,\"custom_fast_math\":false,",
            "\"vendor_is_bit_oracle\":false,\"passed\":true}}"
        ),
        json_escape(SCHEMA),
        json_escape(&metadata.run_identity),
        json_escape(env!("CARGO_PKG_VERSION")),
        TUNING_TABLE_REVISION,
        SHAPES.len() * 4,
        device.compute_capability.0,
        device.compute_capability.1,
        device.multiprocessor_count,
        json_escape(&metadata.nvrtc_target),
        device.driver.api_version,
        device.driver.build_sources,
        json_escape(&digest_hex(&device.driver.build_digest)),
        compiler.nvrtc_version.0,
        compiler.nvrtc_version.1,
        json_escape(compiler.target.as_str()),
        json_escape(&digest_hex(&compiler.source_digest)),
        json_escape(&digest_hex(&compiler.invocation_digest)),
        json_escape(&digest_hex(&compiler.header_manifest_digest)),
        json_escape(&digest_hex(&compiler.nvrtc_library_domain)),
        compiler.nvrtc_library_known,
        json_escape(artifact_kind_name(compiler.output_kind)),
        compiler.composer_revision,
        compiler.compiler_revision,
        compiler.numeric_abi_revision,
        compiler.schedule_revision,
        json_escape(artifact_kind_name(artifact.artifact_kind)),
        json_escape(&digest_hex(&artifact.compile_key)),
        json_escape(&digest_hex(&artifact.artifact_digest)),
        json_escape(&metadata.artifact_set_digest),
        json_escape(&metadata.qualification_source_sha256),
        json_escape(&metadata.rust_dispatch_source_sha256),
        json_escape(&metadata.cuda_source_sha256),
        json_escape(&metadata.executable_sha256),
        json_escape(half_symbol(CANDIDATE_TILE, WeightDtype::Bf16)),
        json_escape(half_symbol(CANDIDATE_TILE, WeightDtype::F16)),
        TIMING_WARMUPS,
        WINDOWS_PER_ORDER,
    ))
}

fn production_aggregate_json(label: &str, aggregate: Aggregate) -> String {
    format!(
        concat!(
            "\"{}\":{{\"summaries\":{},\"candidate_p50_us_sum\":{:.9},",
            "\"comparator_p50_us_sum\":{:.9},\"candidate_p95_us_sum\":{:.9},",
            "\"comparator_p95_us_sum\":{:.9},\"p50_ratio\":{:.9},",
            "\"p95_ratio\":{:.9},\"production_safe_win\":{}}}"
        ),
        json_escape(label),
        aggregate.summaries,
        aggregate.candidate_p50_us,
        aggregate.comparator_p50_us,
        aggregate.candidate_p95_us,
        aggregate.comparator_p95_us,
        aggregate.p50_ratio(),
        aggregate.p95_ratio(),
        aggregate.safe_win(),
    )
}

fn vendor_aggregate_json(label: &str, aggregate: Aggregate) -> String {
    let observation = if aggregate.safe_win() {
        "candidate_stable_win"
    } else {
        "no_candidate_stable_win"
    };
    format!(
        concat!(
            "\"{}\":{{\"summaries\":{},\"candidate_p50_us_sum\":{:.9},",
            "\"comparator_p50_us_sum\":{:.9},\"candidate_p95_us_sum\":{:.9},",
            "\"comparator_p95_us_sum\":{:.9},\"p50_ratio\":{:.9},",
            "\"p95_ratio\":{:.9},\"vendor_observation\":\"{}\",",
            "\"retention_effect\":false}}"
        ),
        json_escape(label),
        aggregate.summaries,
        aggregate.candidate_p50_us,
        aggregate.comparator_p50_us,
        aggregate.candidate_p95_us,
        aggregate.comparator_p95_us,
        aggregate.p50_ratio(),
        aggregate.p95_ratio(),
        json_escape(observation),
    )
}

fn emit_cumulative(
    sink: &mut JsonlSink,
    metadata: &RunMetadata,
    cumulative: &CumulativeResults,
) -> Result<(), String> {
    sink.write(format!(
        concat!(
            "{{\"schema\":\"{}\",\"record_type\":\"cumulative_disposition\",",
            "\"arm\":\"candidate\",\"order\":\"both_orders\",",
            "\"path\":\"eager\",\"bias\":\"all_cells\",",
            "\"run_identity_sha256\":\"{}\",{},{},",
            "\"candidate_retained_cells\":{},\"production_retained_cells\":{},",
            "\"policy\":\"per_cell_production_both_orders_strict_p50_p95\",",
            "\"vendor_retention_effect\":false,\"test_failure_on_performance\":false,",
            "\"safety_gates_passed\":true,",
            "\"disposition\":\"{}\"}}"
        ),
        json_escape(SCHEMA),
        json_escape(&metadata.run_identity),
        production_aggregate_json("all_production_auto", cumulative.all_production),
        vendor_aggregate_json("all_fast_cublas", cumulative.all_vendor),
        cumulative.candidate_retained_cells,
        cumulative.production_retained_cells,
        json_escape(cumulative.disposition()),
    ))
}

const fn expected_records_before_completion(cells: usize) -> usize {
    1 + cells * RECORDS_PER_CELL + 1
}

fn require_output_path() -> Result<PathBuf, String> {
    let value = std::env::var_os(OUTPUT_ENV)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{OUTPUT_ENV} must name a new JSONL file"))?;
    let path = PathBuf::from(value);
    if path.exists() {
        return Err(format!(
            "{OUTPUT_ENV} target {path:?} already exists; qualification requires create_new"
        ));
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(format!(
            "qualification sink parent {parent:?} is not a directory"
        ));
    }
    Ok(path)
}

fn require_sm120_nvrtc(version: (i32, i32)) -> Result<(), String> {
    if version >= (12, 8) {
        Ok(())
    } else {
        Err(format!(
            "Fixed SM120 qualification requires NVRTC 12.8 or newer, got {version:?}"
        ))
    }
}

fn require_environment(device: &GpuDevice, metadata: &RunMetadata) -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("Fixed HALF M64N128 qualification requires --release".into());
    }
    if device.compute_capability != (12, 0) || device.multiprocessor_count() != 170 {
        return Err(format!(
            "qualification requires RTX 5090 CC12.0/170SM, got CC{}.{}/{}SM",
            device.compute_capability.0,
            device.compute_capability.1,
            device.multiprocessor_count()
        ));
    }
    require_sm120_nvrtc(metadata.compiler.nvrtc_version)?;
    if metadata.compiler.target.as_str() != metadata.nvrtc_target {
        return Err(format!(
            "loaded Fixed compiler target {} differs from device target {}",
            metadata.compiler.target.as_str(),
            metadata.nvrtc_target
        ));
    }
    if !metadata.compiler.nvrtc_library_known {
        return Err("loaded Fixed compiler has an unknown NVRTC library identity".into());
    }
    if metadata.artifact.module_kind != ModuleKind::Fixed
        || metadata.compiler.output_kind != metadata.artifact.artifact_kind
    {
        return Err(format!(
            "loaded Fixed compiler/artifact identity is incoherent: {:?} / {:?}",
            metadata.compiler.output_kind, metadata.artifact
        ));
    }
    Ok(())
}

#[test]
#[ignore = "requires an idle RTX 5090 CC12.0/170SM with NVRTC 12.8+ and a new JSONL path"]
fn fixed_half_m64n128_vs_production_auto_qualification() -> Result<(), String> {
    let output_path = require_output_path()?;
    let device = GpuDevice::new(0)?;
    let ctx = GpuCtx::new(&device)?;
    let metadata = RunMetadata::new(&ctx, &device)?;
    require_environment(&device, &metadata)?;
    configure_arm(&ctx, Arm::Candidate)?;

    let mut sink = JsonlSink::create_new(output_path)?;
    emit_run_preflight(&mut sink, &metadata)?;
    let mut cumulative = CumulativeResults::default();
    let cells = qualification_cells();
    for cell in cells.iter().copied() {
        eprintln!("Fixed HALF M64N128 qualification cell {}", cell_id(cell));
        let buffers = CellBuffers::new(&ctx, cell)?;
        if !buffers
            .operands(cell, Arm::ProductionAuto)
            .c
            .ptr
            .is_multiple_of(4)
        {
            return Err(format!(
                "{} production output is not pair-store aligned",
                cell_id(cell)
            ));
        }
        run_correctness(&mut sink, &metadata, &ctx, cell, &buffers)?;
        let summaries = run_timing(&mut sink, &metadata, &ctx, cell, &buffers)?;
        let decision = CellRetentionDecision::from_summaries(&summaries);
        emit_cell_retention(&mut sink, &metadata, cell, &summaries, decision)?;
        cumulative.add_cell_decision(decision);
        for summary in summaries {
            cumulative.add(&summary);
        }
    }
    emit_cumulative(&mut sink, &metadata, &cumulative)?;
    sink.finish(expected_records_before_completion(cells.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata_with_run_identity(run_identity: &str) -> RunMetadata {
        let target =
            mamba_rs::mamba_ssm::gpu::kernel_identity::CudaTarget::new("compute_120").unwrap();
        let compiler = CompilerIdentity {
            source_digest: [1; 32],
            invocation_digest: [2; 32],
            header_manifest_digest: [3; 32],
            target,
            nvrtc_version: (13, 2),
            nvrtc_library_domain: [4; 32],
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Cubin,
            composer_revision: 1,
            compiler_revision: 1,
            numeric_abi_revision: 3,
            schedule_revision: 6,
        };
        let artifact = ArtifactIdentity {
            module_kind: ModuleKind::Fixed,
            artifact_kind: ArtifactKind::Cubin,
            compile_key: [5; 32],
            artifact_digest: [6; 32],
        };
        let device = DeviceIdentity {
            compute_capability: (12, 0),
            multiprocessor_count: 170,
            target,
            driver: mamba_rs::mamba_ssm::gpu::kernel_identity::DriverIdentity {
                api_version: 13_200,
                build_sources: 1,
                build_digest: [7; 32],
            },
        };
        RunMetadata {
            run_identity: run_identity.to_owned(),
            compiler,
            artifact,
            artifact_set_digest: "set".to_owned(),
            device,
            nvrtc_target: "compute_120".to_owned(),
            qualification_source_sha256: "qualification".to_owned(),
            rust_dispatch_source_sha256: "dispatch".to_owned(),
            cuda_source_sha256: "cuda".to_owned(),
            executable_sha256: "executable".to_owned(),
        }
    }

    fn first_cell() -> QualCell {
        qualification_cells()
            .into_iter()
            .next()
            .expect("qualification inventory")
    }

    fn summary(kind: ComparisonKind, order: PairOrder, ratios: (f64, f64)) -> PairSummary {
        PairSummary {
            kind,
            order,
            candidate_p50_us: ratios.0,
            candidate_p95_us: ratios.1,
            comparator_p50_us: 1.0,
            comparator_p95_us: 1.0,
            ratio_p50: ratios.0,
            ratio_p95: ratios.1,
        }
    }

    fn exact_safe_candidate_snapshot() -> PhysicalSnapshot {
        PhysicalSnapshot {
            symbol: half_symbol(CANDIDATE_TILE, WeightDtype::Bf16).to_owned(),
            grid: (144, 1, 1),
            block: (256, 1, 1),
            dynamic_shared_bytes: 49_280,
            registers: 128,
            local_bytes: 0,
            static_shared_bytes: 0,
            max_dynamic_shared_bytes: 49_280,
            preferred_shared_carveout: -1,
            occupancy_blocks_per_sm: 2,
        }
    }

    fn exact_safe_production_snapshot() -> PhysicalSnapshot {
        PhysicalSnapshot {
            symbol: half_symbol(InferenceSm120HalfTile::M64N64Bk64S2, WeightDtype::Bf16).to_owned(),
            grid: (288, 1, 1),
            block: (128, 1, 1),
            dynamic_shared_bytes: 32_896,
            registers: 128,
            local_bytes: 0,
            static_shared_bytes: 0,
            max_dynamic_shared_bytes: 32_896,
            preferred_shared_carveout: -1,
            occupancy_blocks_per_sm: 2,
        }
    }

    #[test]
    fn inventory_covers_twelve_shapes_two_dtypes_and_two_bias_states() {
        let cells = qualification_cells();
        assert_eq!(cells.len(), 48);
        for (_, shape) in SHAPES {
            for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                for bias in [BiasKind::None, BiasKind::Synthesized] {
                    assert_eq!(
                        cells
                            .iter()
                            .filter(|cell| {
                                cell.shape == shape && cell.dtype == dtype && cell.bias == bias
                            })
                            .count(),
                        1,
                    );
                }
            }
        }
    }

    #[test]
    fn sm120_qualification_accepts_every_supported_nvrtc_release() {
        for version in [(12, 8), (13, 0), (13, 1), (13, 2)] {
            assert!(require_sm120_nvrtc(version).is_ok(), "{version:?}");
        }
        assert!(require_sm120_nvrtc((12, 7)).is_err());
        assert!(require_sm120_nvrtc((0, 0)).is_err());
    }

    #[test]
    fn physical_gate_accepts_exact_safe_candidate_and_production() {
        let compiler = metadata_with_run_identity("safe").compiler;
        assert!(
            validate_physical(
                first_cell(),
                compiler,
                Arm::Candidate,
                &exact_safe_candidate_snapshot(),
            )
            .is_ok()
        );
        assert!(
            validate_physical(
                first_cell(),
                compiler,
                Arm::ProductionAuto,
                &exact_safe_production_snapshot(),
            )
            .is_ok()
        );
    }

    #[test]
    fn physical_gate_rejects_each_resource_regression() {
        let compiler = metadata_with_run_identity("unsafe").compiler;
        let mut snapshot = exact_safe_candidate_snapshot();
        snapshot.local_bytes = 4;
        assert!(validate_physical(first_cell(), compiler, Arm::Candidate, &snapshot).is_err());

        snapshot = exact_safe_candidate_snapshot();
        snapshot.static_shared_bytes = 4;
        assert!(validate_physical(first_cell(), compiler, Arm::Candidate, &snapshot).is_err());

        snapshot = exact_safe_candidate_snapshot();
        snapshot.max_dynamic_shared_bytes = 49_279;
        assert!(validate_physical(first_cell(), compiler, Arm::Candidate, &snapshot).is_err());

        snapshot = exact_safe_candidate_snapshot();
        snapshot.occupancy_blocks_per_sm = 1;
        assert!(validate_physical(first_cell(), compiler, Arm::Candidate, &snapshot).is_err());

        snapshot = exact_safe_candidate_snapshot();
        snapshot.registers = 129;
        assert!(validate_physical(first_cell(), compiler, Arm::Candidate, &snapshot).is_err());

        snapshot = exact_safe_candidate_snapshot();
        snapshot.preferred_shared_carveout = 0;
        assert!(validate_physical(first_cell(), compiler, Arm::Candidate, &snapshot).is_err());
    }

    #[test]
    fn production_auto_accepts_the_promoted_candidate_route() {
        let compiler = metadata_with_run_identity("auto").compiler;
        let snapshot = exact_safe_candidate_snapshot();
        assert!(validate_physical(first_cell(), compiler, Arm::ProductionAuto, &snapshot).is_ok());
    }

    #[test]
    fn vendor_observation_never_gates_a_production_safe_win() {
        let mut summaries = Vec::new();
        for order in [PairOrder::CandidateFirst, PairOrder::ComparatorFirst] {
            summaries.push(summary(ComparisonKind::ProductionAuto, order, (0.8, 0.9)));
            summaries.push(summary(ComparisonKind::FastCublas, order, (1.2, 1.3)));
        }
        let decision = CellRetentionDecision::from_summaries(&summaries);
        assert!(decision.production_safe_win());
        assert_eq!(decision.vendor_observation(), "no_candidate_stable_win");
        assert_eq!(decision.retained_arm(), Arm::Candidate);
    }

    #[test]
    fn retention_requires_strict_p50_and_p95_in_both_orders() {
        let one_order = [summary(
            ComparisonKind::ProductionAuto,
            PairOrder::CandidateFirst,
            (0.8, 0.9),
        )];
        assert!(!CellRetentionDecision::from_summaries(&one_order).production_safe_win());

        let p95_tie = [
            summary(
                ComparisonKind::ProductionAuto,
                PairOrder::CandidateFirst,
                (0.8, 0.9),
            ),
            summary(
                ComparisonKind::ProductionAuto,
                PairOrder::ComparatorFirst,
                (0.8, 1.0),
            ),
        ];
        assert!(!CellRetentionDecision::from_summaries(&p95_tie).production_safe_win());
    }

    #[test]
    fn nearest_rank_uses_all_101_windows() {
        let values = (1..=101).map(|value| value as f64).collect::<Vec<_>>();
        assert_eq!(nearest_rank(&values, 0.50).unwrap(), 51.0);
        assert_eq!(nearest_rank(&values, 0.95).unwrap(), 96.0);
    }

    #[test]
    fn telemetry_parser_enforces_complete_positive_samples() {
        let (gpu_util, memory_util, used_mib, snapshot) =
            parse_telemetry("1, 0, 2048, 2400, 48, P0").unwrap();
        assert_eq!((gpu_util, memory_util), (1, 0));
        assert_eq!(used_mib, 2048);
        assert!(snapshot.contains("used=2048MiB"));
        assert!(telemetry_is_quiet(1, 1));
        assert!(!telemetry_is_quiet(2, 1));
        assert!(!telemetry_is_quiet(1, 2));
        assert!(parse_telemetry("1,0,2048,0,48,P0").is_err());
        assert!(parse_telemetry("1,0,2048,2400,0,P0").is_err());
        assert!(parse_telemetry("1,0,2048").is_err());
    }

    #[test]
    fn compiled_source_hashes_match_the_qualification_checkout() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        for (path, compiled) in [
            (
                manifest.join("tests/gemm_bi_inference_half_m64n128_qualification.rs"),
                QUALIFICATION_SOURCE,
            ),
            (
                manifest.join("src/mamba_ssm/gpu/gemm_bi_inference.rs"),
                RUST_DISPATCH_SOURCE,
            ),
            (
                manifest.join("kernels/gemm_bi_inference/sm120_tma.cu"),
                CUDA_SOURCE,
            ),
        ] {
            assert_eq!(
                verify_compiled_source(&path, compiled).unwrap().len(),
                64,
                "{path:?}"
            );
        }
    }

    #[test]
    fn completion_count_covers_every_cell_cohort_and_summary() {
        assert_eq!(RECORDS_PER_CELL, 422);
        assert_eq!(expected_records_before_completion(48), 20_258);
    }

    #[test]
    fn json_escape_preserves_jsonl_boundaries_and_control_characters() {
        assert_eq!(
            json_escape("a\"b\\c\nd\re\tf\u{1}"),
            "a\\\"b\\\\c\\nd\\re\\tf\\u0001"
        );
    }

    #[test]
    fn cell_json_carries_dtype_bias_and_escaped_identity() {
        let metadata = metadata_with_run_identity("run\"\\\n");
        let cell = QualCell {
            shape_name: "shape\tname",
            shape: InferenceShape { m: 1, k: 2, n: 3 },
            dtype: WeightDtype::F16,
            bias: BiasKind::Synthesized,
        };
        assert_eq!(
            cell_json(&metadata, cell),
            r#""schema":"MambaBiFixedHalfM64N128QualificationV1","run_identity_sha256":"run\"\\\n","cell":"shape\tname_f16_synthesized","shape_name":"shape\tname","shape":{"m":1,"k":2,"n":3},"dtype":"f16","bias":"synthesized""#
        );
    }
}
