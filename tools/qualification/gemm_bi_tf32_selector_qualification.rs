#![cfg(feature = "cuda")]

#[path = "../../tests/common/digest.rs"]
mod digest;

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write as _};
use std::path::{Path as FsPath, PathBuf};

use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::blas::{gpu_gemm_bi_backward_dw_grad, gpu_gemm_bi_backward_dx_raw};
use mamba_rs::mamba_ssm::gpu::buffers::{GpuBuffer, GradSlice};
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    PhysicalQualificationRequest, PhysicalQualificationRoute, QualifiedPhysicalLaunch,
    presize_physical_qualification_suite, qualify_physical_launch,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    ModuleKind, PolicyDtype, ResolvedGemmOp, digest_hex,
};
use sha2::{Digest as _, Sha256};

const WINDOWS_PER_ORDER: usize = 101;
const REPEATS: usize = 10;
const WARMUPS: usize = 128;
const TARGET_WINDOW_MS: f64 = 5.0;
const OUTPUT_ENV: &str = "MAMBA_RS_TF32_SELECTOR_JSONL";
const EXPECTED_TILE: (u32, u32) = (16, 32);
const RECORDS_PER_CELL_BEFORE_COMPLETION: usize = 413;
const EXPECTED_RECORDS_BEFORE_COMPLETION: usize =
    1 + CELLS.len() * RECORDS_PER_CELL_BEFORE_COMPLETION;

#[derive(Clone, Copy, Debug)]
struct Cell {
    id: &'static str,
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
    symbol: &'static str,
}

const CELLS: [Cell; 5] = [
    Cell {
        id: "tn_r49_c129_k65",
        op: ResolvedGemmOp::Tn,
        dims: (65, 49, 129),
        symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4",
    },
    Cell {
        id: "tn_r65_c129_k49",
        op: ResolvedGemmOp::Tn,
        dims: (49, 65, 129),
        symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4",
    },
    Cell {
        id: "tn_r131_c100_k129",
        op: ResolvedGemmOp::Tn,
        dims: (129, 131, 100),
        symbol: "gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4",
    },
    Cell {
        id: "nt_r49_c65_k129",
        op: ResolvedGemmOp::Nt,
        dims: (49, 65, 129),
        symbol: "gemm_bi_nt_sm80_mma_tf32_v1_m16n32_bk32_s4",
    },
    Cell {
        id: "nt_r65_c49_k129",
        op: ResolvedGemmOp::Nt,
        dims: (65, 49, 129),
        symbol: "gemm_bi_nt_sm80_mma_tf32_v1_m16n32_bk32_s4",
    },
];

#[derive(Clone, Copy, Debug)]
enum Arm {
    Candidate,
    Incumbent,
}

impl Arm {
    const fn name(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Incumbent => "incumbent",
        }
    }

    const fn policy(self) -> F32TriadPolicy {
        match self {
            Self::Candidate => F32TriadPolicy::AllowDeterministicTf32V1,
            Self::Incumbent => F32TriadPolicy::ExactScalarFmaV1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ExecutionPath {
    Eager,
    Graph,
}

impl ExecutionPath {
    const fn name(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Graph => "graph",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ComparatorOrder {
    CandidateThenIncumbent,
    IncumbentThenCandidate,
}

impl ComparatorOrder {
    const fn name(self) -> &'static str {
        match self {
            Self::CandidateThenIncumbent => "candidate_then_incumbent",
            Self::IncumbentThenCandidate => "incumbent_then_candidate",
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
    fn create_from_env() -> Result<Self, String> {
        let value = std::env::var_os(OUTPUT_ENV)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("{OUTPUT_ENV} must name a new JSONL file"))?;
        let path = PathBuf::from(value);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| format!("create dedicated TF32 selector sink {path:?}: {error}"))?;
        if !file
            .metadata()
            .map_err(|error| format!("inspect TF32 selector sink {path:?}: {error}"))?
            .file_type()
            .is_file()
        {
            return Err(format!("TF32 selector sink {path:?} is not a regular file"));
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
            .map_err(|error| format!("write TF32 selector sink {:?}: {error}", self.path))?;
        self.digest.update(&bytes);
        self.records += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<(), String> {
        let prior_records = self.records;
        if prior_records != EXPECTED_RECORDS_BEFORE_COMPLETION {
            return Err(format!(
                "incomplete TF32 selector JSONL: got {prior_records} records before completion, expected {EXPECTED_RECORDS_BEFORE_COMPLETION}"
            ));
        }
        let prior_digest = format!("{:x}", self.digest.clone().finalize());
        self.write(format!(
            concat!(
                "{{\"schema\":\"MambaBiTf32SelectorQualificationV1\",",
                "\"record_type\":\"completion\",",
                "\"records_before_completion\":{},",
                "\"expected_records_before_completion\":{},",
                "\"expected_records_total\":{},",
                "\"content_sha256_before_completion\":\"{}\",",
                "\"complete\":true}}"
            ),
            prior_records,
            EXPECTED_RECORDS_BEFORE_COMPLETION,
            EXPECTED_RECORDS_BEFORE_COMPLETION + 1,
            prior_digest,
        ))?;
        self.writer
            .flush()
            .map_err(|error| format!("flush TF32 selector sink {:?}: {error}", self.path))?;
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("sync TF32 selector sink {:?}: {error}", self.path))?;
        eprintln!(
            "TF32 selector qualification wrote {} JSONL records to {:?}",
            self.records, self.path
        );
        Ok(())
    }
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

fn optional_json_string(value: Option<&str>) -> String {
    value.map_or_else(
        || "null".to_owned(),
        |value| format!("\"{}\"", json_escape(value)),
    )
}

fn optional_json_tile(value: Option<(u32, u32)>) -> String {
    value.map_or_else(|| "null".to_owned(), |(m, n)| format!("[{m},{n}]"))
}

fn configure_context(device: &GpuDevice, policy: F32TriadPolicy) -> Result<GpuCtx, String> {
    let ctx = GpuCtx::new(device)?;
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(policy);
    Ok(ctx)
}

fn qualification_request(cell: Cell, arm: Arm) -> PhysicalQualificationRequest {
    PhysicalQualificationRequest::contiguous(
        cell.op,
        cell.dims,
        PhysicalQualificationRoute::F32Policy(arm.policy()),
    )
}

fn assert_physical_route(
    sink: &mut JsonlSink,
    cell: Cell,
    arm: Arm,
    qualified: &QualifiedPhysicalLaunch<'_>,
) -> Result<(), String> {
    let evidence = qualified.evidence();
    if !evidence.eager_graph_equal() {
        return Err(format!(
            "{} {} eager/graph route mismatch",
            cell.id,
            arm.name()
        ));
    }
    if evidence.launch_count() == 0 {
        return Err(format!("{} {} has no physical launch", cell.id, arm.name()));
    }
    if evidence.uniform_execution_dtype() != Some(PolicyDtype::F32) {
        return Err(format!(
            "{} {} physical route changed execution dtype",
            cell.id,
            arm.name()
        ));
    }
    if matches!(arm, Arm::Candidate)
        && (evidence.launch_count() != 1
            || evidence.single_launch_symbol() != Some(cell.symbol)
            || evidence.single_launch_tile() != Some(EXPECTED_TILE)
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadSm80))
    {
        return Err(format!(
            "{} candidate physical route differs from M16N32/S4: {:?}",
            cell.id,
            evidence.nodes()
        ));
    }
    if matches!(arm, Arm::Incumbent)
        && (evidence.single_launch_symbol() == Some(cell.symbol)
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadScalar))
    {
        return Err(format!(
            "{} exact incumbent did not remain in TriadScalar: {:?}",
            cell.id,
            evidence.nodes()
        ));
    }
    sink.write(format!(
        concat!(
            "{{\"schema\":\"MambaBiTf32SelectorQualificationV1\",",
            "\"record_type\":\"physical_preflight\",",
            "\"cell\":\"{}\",\"arm\":\"{}\",\"op\":\"{}\",",
            "\"dims\":[{},{},{}],\"evidence_scope\":\"{}\",",
            "\"eager_graph_equal\":{},\"launch_count\":{},",
            "\"launch_digest\":\"{}\",\"request_digest\":\"{}\",",
            "\"symbol\":{},\"tile\":{},\"module_kind\":\"{}\",",
            "\"execution_dtype\":\"f32\",\"passed\":true}}"
        ),
        json_escape(cell.id),
        arm.name(),
        op_name(cell.op),
        cell.dims.0,
        cell.dims.1,
        cell.dims.2,
        json_escape(evidence.evidence_scope()),
        evidence.eager_graph_equal(),
        evidence.launch_count(),
        digest_hex(&evidence.launch_digest()),
        digest_hex(&evidence.request_identity_digest()),
        optional_json_string(evidence.single_launch_symbol()),
        optional_json_tile(evidence.single_launch_tile()),
        match arm {
            Arm::Candidate => "triad_sm80",
            Arm::Incumbent => "triad_scalar",
        },
    ))
}

struct BitBuffers {
    output: GpuBuffer,
    a: GpuBuffer,
    b: GpuBuffer,
    seed: Vec<f32>,
}

impl BitBuffers {
    fn new(ctx: &GpuCtx, cell: Cell) -> Result<Self, String> {
        let (m, k, n) = cell.dims;
        let (output_len, a_len, b_len) = match cell.op {
            ResolvedGemmOp::Tn => (k * n, m * k, m * n),
            ResolvedGemmOp::Nt => (m * k, m * n, k * n),
            ResolvedGemmOp::Nn => return Err("selector qualification admits TN/NT only".into()),
        };
        let seed = seeded_values(output_len, 0x91);
        Ok(Self {
            output: GpuBuffer::from_cpu(&ctx.stream, &seed)?,
            a: GpuBuffer::from_cpu(&ctx.stream, &seeded_values(a_len, 0x2d))?,
            b: GpuBuffer::from_cpu(&ctx.stream, &seeded_values(b_len, 0x67))?,
            seed,
        })
    }

    fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
        self.output.upload(&ctx.stream, &self.seed)
    }

    fn launch(&mut self, ctx: &GpuCtx, cell: Cell) -> Result<(), String> {
        let (m, k, n) = cell.dims;
        match cell.op {
            ResolvedGemmOp::Tn => gpu_gemm_bi_backward_dw_grad(
                ctx,
                &GradSlice::from_raw(self.output.cached_ptr(), k * n),
                &self.b,
                &self.a,
                m,
                k,
                n,
            ),
            ResolvedGemmOp::Nt => gpu_gemm_bi_backward_dx_raw(
                ctx,
                &mut self.output,
                &self.a,
                self.b.cached_ptr(),
                m,
                k,
                n,
            ),
            ResolvedGemmOp::Nn => Err("selector qualification admits TN/NT only".into()),
        }
    }

    fn values(&self, ctx: &GpuCtx) -> Result<Vec<f32>, String> {
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize selector bit gate: {error:?}"))?;
        self.output.to_cpu(&ctx.stream)
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

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn run_bit_gate(
    sink: &mut JsonlSink,
    device: &GpuDevice,
    cell: Cell,
    arm: Arm,
) -> Result<(), String> {
    let ctx = configure_context(device, arm.policy())?;
    let mut buffers = BitBuffers::new(&ctx, cell)?;

    buffers.reset(&ctx)?;
    buffers.launch(&ctx, cell)?;
    let eager = buffers.values(&ctx)?;
    let eager_bits = bits(&eager);
    for repeat in 0..REPEATS {
        buffers.reset(&ctx)?;
        buffers.launch(&ctx, cell)?;
        let actual = bits(&buffers.values(&ctx)?);
        if actual != eager_bits {
            return Err(format!(
                "{} {} eager repeat {repeat} changed bits",
                cell.id,
                arm.name()
            ));
        }
    }

    buffers.reset(&ctx)?;
    let graph = unsafe { capture_into_graph(&ctx.stream, || buffers.launch(&ctx, cell)) }?;
    for replay in 0..REPEATS {
        buffers.reset(&ctx)?;
        graph
            .launch()
            .map_err(|error| format!("{} graph replay {replay}: {error:?}", cell.id))?;
        let actual = bits(&buffers.values(&ctx)?);
        if actual != eager_bits {
            return Err(format!(
                "{} {} graph replay {replay} changed bits",
                cell.id,
                arm.name()
            ));
        }
    }

    sink.write(format!(
        concat!(
            "{{\"schema\":\"MambaBiTf32SelectorQualificationV1\",",
            "\"record_type\":\"bit_gate\",\"cell\":\"{}\",\"arm\":\"{}\",",
            "\"seeded_allocations\":true,\"eager_repeats\":{},",
            "\"warm_graph_replays\":{},\"output_fnv1a64\":\"{:016x}\",",
            "\"exact_eager_repeat_bits\":true,\"exact_eager_graph_bits\":true,",
            "\"passed\":true}}"
        ),
        json_escape(cell.id),
        arm.name(),
        REPEATS,
        REPEATS,
        digest::fnv1a_f32(&eager),
    ))
}

fn measure(
    qualified: &mut QualifiedPhysicalLaunch<'_>,
    ctx: &GpuCtx,
    path: ExecutionPath,
    iterations: usize,
) -> Result<f64, String> {
    if iterations == 0 {
        return Err("selector timing iterations must be positive".into());
    }
    let elapsed = match path {
        ExecutionPath::Eager => qualified.measure_eager_window_ms(ctx, iterations)?,
        ExecutionPath::Graph => qualified.measure_graph_window_ms(ctx, iterations)?,
    };
    let per_call_us = elapsed * 1000.0 / iterations as f64;
    if !per_call_us.is_finite() || per_call_us <= 0.0 {
        return Err(format!("invalid selector timing sample {per_call_us}"));
    }
    Ok(per_call_us)
}

fn calibrate(
    qualified: &mut QualifiedPhysicalLaunch<'_>,
    ctx: &GpuCtx,
    path: ExecutionPath,
) -> Result<usize, String> {
    let probe_iterations = 16;
    let per_call_us = measure(qualified, ctx, path, probe_iterations)?;
    Ok(((TARGET_WINDOW_MS * 1000.0 / per_call_us).ceil() as usize).clamp(1, 4096))
}

fn percentile(values: &[f64], fraction: f64) -> Result<f64, String> {
    if values.len() != WINDOWS_PER_ORDER
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("percentile requires exactly 101 positive finite samples".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Ok(sorted[index])
}

fn run_paired_path(
    sink: &mut JsonlSink,
    cell: Cell,
    path: ExecutionPath,
    order: ComparatorOrder,
    candidate: (&GpuCtx, &mut QualifiedPhysicalLaunch<'_>, usize),
    incumbent: (&GpuCtx, &mut QualifiedPhysicalLaunch<'_>, usize),
) -> Result<(), String> {
    let (candidate_ctx, candidate, candidate_iterations) = candidate;
    let (incumbent_ctx, incumbent, incumbent_iterations) = incumbent;
    measure(candidate, candidate_ctx, path, WARMUPS)?;
    measure(incumbent, incumbent_ctx, path, WARMUPS)?;

    let mut candidate_us = Vec::with_capacity(WINDOWS_PER_ORDER);
    let mut incumbent_us = Vec::with_capacity(WINDOWS_PER_ORDER);
    let mut ratios = Vec::with_capacity(WINDOWS_PER_ORDER);
    for window in 0..WINDOWS_PER_ORDER {
        let (candidate_sample, incumbent_sample) = match order {
            ComparatorOrder::CandidateThenIncumbent => (
                measure(candidate, candidate_ctx, path, candidate_iterations)?,
                measure(incumbent, incumbent_ctx, path, incumbent_iterations)?,
            ),
            ComparatorOrder::IncumbentThenCandidate => {
                let incumbent_sample =
                    measure(incumbent, incumbent_ctx, path, incumbent_iterations)?;
                let candidate_sample =
                    measure(candidate, candidate_ctx, path, candidate_iterations)?;
                (candidate_sample, incumbent_sample)
            }
        };
        let ratio = candidate_sample / incumbent_sample;
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err(format!(
                "{} invalid paired ratio at window {window}",
                cell.id
            ));
        }
        candidate_us.push(candidate_sample);
        incumbent_us.push(incumbent_sample);
        ratios.push(ratio);
        sink.write(format!(
            concat!(
                "{{\"schema\":\"MambaBiTf32SelectorQualificationV1\",",
                "\"record_type\":\"paired_window\",\"cell\":\"{}\",",
                "\"path\":\"{}\",\"comparator_order\":\"{}\",\"window\":{},",
                "\"candidate_iterations\":{},\"incumbent_iterations\":{},",
                "\"candidate_us\":{},\"incumbent_us\":{},",
                "\"candidate_over_incumbent\":{}}}"
            ),
            json_escape(cell.id),
            path.name(),
            order.name(),
            window,
            candidate_iterations,
            incumbent_iterations,
            candidate_sample,
            incumbent_sample,
            ratio,
        ))?;
    }

    sink.write(format!(
        concat!(
            "{{\"schema\":\"MambaBiTf32SelectorQualificationV1\",",
            "\"record_type\":\"paired_summary\",\"cell\":\"{}\",",
            "\"path\":\"{}\",\"comparator_order\":\"{}\",\"windows\":{},",
            "\"candidate_p05_us\":{},\"candidate_p50_us\":{},\"candidate_p95_us\":{},",
            "\"incumbent_p05_us\":{},\"incumbent_p50_us\":{},\"incumbent_p95_us\":{},",
            "\"ratio_p05\":{},\"ratio_p50\":{},\"ratio_p95\":{},\"passed\":true}}"
        ),
        json_escape(cell.id),
        path.name(),
        order.name(),
        WINDOWS_PER_ORDER,
        percentile(&candidate_us, 0.05)?,
        percentile(&candidate_us, 0.50)?,
        percentile(&candidate_us, 0.95)?,
        percentile(&incumbent_us, 0.05)?,
        percentile(&incumbent_us, 0.50)?,
        percentile(&incumbent_us, 0.95)?,
        percentile(&ratios, 0.05)?,
        percentile(&ratios, 0.50)?,
        percentile(&ratios, 0.95)?,
    ))
}

fn run_paired_cell(sink: &mut JsonlSink, device: &GpuDevice, cell: Cell) -> Result<(), String> {
    let candidate_ctx = configure_context(device, Arm::Candidate.policy())?;
    let incumbent_ctx = configure_context(device, Arm::Incumbent.policy())?;
    let candidate_request = qualification_request(cell, Arm::Candidate);
    let incumbent_request = qualification_request(cell, Arm::Incumbent);
    presize_physical_qualification_suite(&candidate_ctx, &[candidate_request])?;
    presize_physical_qualification_suite(&incumbent_ctx, &[incumbent_request])?;
    let mut candidate = qualify_physical_launch(&candidate_ctx, candidate_request)?;
    let mut incumbent = qualify_physical_launch(&incumbent_ctx, incumbent_request)?;
    candidate.validate_timed_request(&candidate_ctx, candidate_request)?;
    incumbent.validate_timed_request(&incumbent_ctx, incumbent_request)?;
    assert_physical_route(sink, cell, Arm::Candidate, &candidate)?;
    assert_physical_route(sink, cell, Arm::Incumbent, &incumbent)?;

    let candidate_eager = calibrate(&mut candidate, &candidate_ctx, ExecutionPath::Eager)?;
    let candidate_graph = calibrate(&mut candidate, &candidate_ctx, ExecutionPath::Graph)?;
    let incumbent_eager = calibrate(&mut incumbent, &incumbent_ctx, ExecutionPath::Eager)?;
    let incumbent_graph = calibrate(&mut incumbent, &incumbent_ctx, ExecutionPath::Graph)?;
    sink.write(format!(
        concat!(
            "{{\"schema\":\"MambaBiTf32SelectorQualificationV1\",",
            "\"record_type\":\"calibration\",\"cell\":\"{}\",",
            "\"target_window_ms\":{},\"candidate_eager_iterations\":{},",
            "\"candidate_graph_iterations\":{},\"incumbent_eager_iterations\":{},",
            "\"incumbent_graph_iterations\":{}}}"
        ),
        json_escape(cell.id),
        TARGET_WINDOW_MS,
        candidate_eager,
        candidate_graph,
        incumbent_eager,
        incumbent_graph,
    ))?;

    for path in [ExecutionPath::Eager, ExecutionPath::Graph] {
        let (candidate_iterations, incumbent_iterations) = match path {
            ExecutionPath::Eager => (candidate_eager, incumbent_eager),
            ExecutionPath::Graph => (candidate_graph, incumbent_graph),
        };
        for order in [
            ComparatorOrder::CandidateThenIncumbent,
            ComparatorOrder::IncumbentThenCandidate,
        ] {
            run_paired_path(
                sink,
                cell,
                path,
                order,
                (&candidate_ctx, &mut candidate, candidate_iterations),
                (&incumbent_ctx, &mut incumbent, incumbent_iterations),
            )?;
        }
    }
    Ok(())
}

fn op_name(op: ResolvedGemmOp) -> &'static str {
    match op {
        ResolvedGemmOp::Nn => "nn",
        ResolvedGemmOp::Tn => "tn",
        ResolvedGemmOp::Nt => "nt",
    }
}

fn require_release_profile() -> Result<(), String> {
    if cfg!(debug_assertions) {
        Err("TF32 selector qualification requires --release".into())
    } else {
        Ok(())
    }
}

fn require_new_sink_parent(path: &FsPath) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if parent.is_some_and(|parent| !parent.is_dir()) {
        return Err(format!(
            "TF32 selector sink parent {parent:?} is not a directory"
        ));
    }
    Ok(())
}

#[test]
#[ignore = "requires an idle 142-SM CC8.9 GPU and writes 101-window paired JSONL"]
fn sm89_tf32_selector_five_cell_qualification() -> Result<(), String> {
    require_release_profile()?;
    let sink_path = std::env::var_os(OUTPUT_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| format!("{OUTPUT_ENV} must name a new JSONL file"))?;
    require_new_sink_parent(&sink_path)?;
    let mut sink = JsonlSink::create_from_env()?;
    let device = GpuDevice::new(0)?;
    if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
        return Err(format!(
            "selector qualification requires CC8.9/142SM, got {:?}/{}SM",
            device.compute_capability,
            device.multiprocessor_count()
        ));
    }
    sink.write(format!(
        concat!(
            "{{\"schema\":\"MambaBiTf32SelectorQualificationV1\",",
            "\"record_type\":\"run_preflight\",\"cc\":\"8.9\",",
            "\"multiprocessor_count\":142,\"cells\":{},\"windows_per_order\":{},",
            "\"orders\":[\"candidate_then_incumbent\",\"incumbent_then_candidate\"],",
            "\"paths\":[\"eager\",\"graph\"],\"release_build\":true,",
            "\"passed\":true}}"
        ),
        CELLS.len(),
        WINDOWS_PER_ORDER,
    ))?;

    for cell in CELLS {
        eprintln!("TF32 selector non-timed bit gates: {}", cell.id);
        run_bit_gate(&mut sink, &device, cell, Arm::Candidate)?;
        run_bit_gate(&mut sink, &device, cell, Arm::Incumbent)?;
        eprintln!("TF32 selector paired timing: {}", cell.id);
        run_paired_cell(&mut sink, &device, cell)?;
    }
    sink.finish()
}

#[test]
fn selector_qualification_percentiles_use_nearest_rank_over_101_windows() {
    let values = (1..=WINDOWS_PER_ORDER)
        .map(|value| value as f64)
        .collect::<Vec<_>>();
    assert_eq!(percentile(&values, 0.05).unwrap(), 6.0);
    assert_eq!(percentile(&values, 0.50).unwrap(), 51.0);
    assert_eq!(percentile(&values, 0.95).unwrap(), 96.0);
}

#[test]
fn selector_qualification_json_escaping_preserves_jsonl_boundaries() {
    assert_eq!(
        json_escape("a\"b\\c\nd\re\tf\u{1}"),
        "a\\\"b\\\\c\\nd\\re\\tf\\u0001"
    );
    assert_eq!(optional_json_string(None), "null");
    assert_eq!(optional_json_tile(Some((16, 32))), "[16,32]");
    assert_eq!(EXPECTED_RECORDS_BEFORE_COMPLETION, 2_066);
}
