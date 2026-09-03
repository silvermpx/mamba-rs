#![cfg(feature = "cuda")]

mod common;

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write as _};
use std::path::PathBuf;

use common::gpu_quiet::QuietGpu;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    PhysicalQualificationF32Epilogue, PhysicalQualificationRequest, PhysicalQualificationRoute,
    QualifiedPhysicalLaunch, Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages,
    Tf32PortableTile, Tf32QualifiedModule, presize_physical_qualification_suite,
    qualify_physical_launch, tf32_route_specs,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    GemmRouteIdentity, ModuleKind, ResolvedGemmOp, digest_hex,
};
use sha2::{Digest as _, Sha256};

const ENABLE_ENV: &str = "MAMBA_RS_SM120_TF32_SELECTOR_QUALIFICATION";
const OUTPUT_ENV: &str = "MAMBA_RS_SM120_TF32_SELECTOR_JSONL";
const DISCOVERY_WINDOWS: usize = 21;
const FINAL_WINDOWS: usize = 101;
const TARGET_WINDOW_MS: f64 = 5.0;
const WARMUP_LAUNCHES: usize = 128;
const CALIBRATION_PROBE_LAUNCHES: usize = 16;
const MAX_WINDOW_LAUNCHES: usize = 1_000_000;
const MIN_MEDIAN_SPEEDUP: f64 = 1.01;
const MIN_P05_SPEEDUP: f64 = 1.0;
const CORPUS_SALT: u64 = 0x1205_32a1;
const SCHEMA: &str = "MambaBiSm120Tf32SelectorQualificationV1";
const FIXED_SPLITK_SCRATCH_ELEMENTS: usize = 1 << 23;
const DISPATCH_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs");
const TEST_SOURCE: &str = include_str!("gemm_bi_sm120_tf32_selector_qualification.rs");

#[derive(Clone, Copy, Debug, PartialEq)]
struct Cell {
    id: &'static str,
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
    alpha: f32,
    beta: f32,
    bias: bool,
}

impl Cell {
    const fn epilogue(self) -> PhysicalQualificationF32Epilogue {
        PhysicalQualificationF32Epilogue::new(self.alpha, self.beta, self.bias)
    }
}

const CELLS: [Cell; 18] = [
    Cell {
        id: "nn_m64_k384_n1536_bias",
        op: ResolvedGemmOp::Nn,
        dims: (64, 384, 1536),
        alpha: 1.0,
        beta: 0.0,
        bias: true,
    },
    Cell {
        id: "nn_m64_k1536_n384_bias",
        op: ResolvedGemmOp::Nn,
        dims: (64, 1536, 384),
        alpha: 1.0,
        beta: 0.0,
        bias: true,
    },
    Cell {
        id: "nn_m64_k833_n384_bias",
        op: ResolvedGemmOp::Nn,
        dims: (64, 833, 384),
        alpha: 1.0,
        beta: 0.0,
        bias: true,
    },
    Cell {
        id: "nn_m64_k384_n101_bias",
        op: ResolvedGemmOp::Nn,
        dims: (64, 384, 101),
        alpha: 1.0,
        beta: 0.0,
        bias: true,
    },
    Cell {
        id: "nt_m64_k384_n1536",
        op: ResolvedGemmOp::Nt,
        dims: (64, 384, 1536),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "nt_m64_k1536_n384",
        op: ResolvedGemmOp::Nt,
        dims: (64, 1536, 384),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "nn_m64_k256_n256_bias",
        op: ResolvedGemmOp::Nn,
        dims: (64, 256, 256),
        alpha: 1.0,
        beta: 0.0,
        bias: true,
    },
    Cell {
        id: "tn_m64_k384_n1536_beta1",
        op: ResolvedGemmOp::Tn,
        dims: (64, 384, 1536),
        alpha: 1.0,
        beta: 1.0,
        bias: false,
    },
    Cell {
        id: "tn_m64_k1536_n384_beta1",
        op: ResolvedGemmOp::Tn,
        dims: (64, 1536, 384),
        alpha: 1.0,
        beta: 1.0,
        bias: false,
    },
    Cell {
        id: "nn_m4096_k128_n796",
        op: ResolvedGemmOp::Nn,
        dims: (4096, 128, 796),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "nn_m4096_k256_n128",
        op: ResolvedGemmOp::Nn,
        dims: (4096, 256, 128),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "nn_m64_k192_n256_bias",
        op: ResolvedGemmOp::Nn,
        dims: (64, 192, 256),
        alpha: 1.0,
        beta: 0.0,
        bias: true,
    },
    Cell {
        id: "nn_m64_k256_n1024_bias",
        op: ResolvedGemmOp::Nn,
        dims: (64, 256, 1024),
        alpha: 1.0,
        beta: 0.0,
        bias: true,
    },
    Cell {
        id: "nn_m64_k1024_n256_bias",
        op: ResolvedGemmOp::Nn,
        dims: (64, 1024, 256),
        alpha: 1.0,
        beta: 0.0,
        bias: true,
    },
    Cell {
        id: "nn_m10400_k768_n384",
        op: ResolvedGemmOp::Nn,
        dims: (10400, 768, 384),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "tn_m10400_k384_n1536_beta1",
        op: ResolvedGemmOp::Tn,
        dims: (10400, 384, 1536),
        alpha: 1.0,
        beta: 1.0,
        bias: false,
    },
    Cell {
        id: "tn_m10400_k768_n384_beta1",
        op: ResolvedGemmOp::Tn,
        dims: (10400, 768, 384),
        alpha: 1.0,
        beta: 1.0,
        bias: false,
    },
    Cell {
        id: "nt_m10400_k768_n384",
        op: ResolvedGemmOp::Nt,
        dims: (10400, 768, 384),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
];

const PROJECTION_CELLS: [Cell; 21] = [
    Cell {
        id: "nn_d768_in_proj",
        op: ResolvedGemmOp::Nn,
        dims: (2048, 768, 3072),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "tn_d768_in_proj",
        op: ResolvedGemmOp::Tn,
        dims: (2048, 768, 3072),
        alpha: 1.0,
        beta: 1.0,
        bias: false,
    },
    Cell {
        id: "nt_d768_in_proj",
        op: ResolvedGemmOp::Nt,
        dims: (2048, 768, 3072),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "nn_d768_out_proj",
        op: ResolvedGemmOp::Nn,
        dims: (2048, 1536, 768),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "tn_d768_out_proj",
        op: ResolvedGemmOp::Tn,
        dims: (2048, 1536, 768),
        alpha: 1.0,
        beta: 1.0,
        bias: false,
    },
    Cell {
        id: "nt_d768_out_proj",
        op: ResolvedGemmOp::Nt,
        dims: (2048, 1536, 768),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "nn_prism_in_proj",
        op: ResolvedGemmOp::Nn,
        dims: (4621, 384, 1928),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "tn_prism_in_proj",
        op: ResolvedGemmOp::Tn,
        dims: (4621, 384, 1928),
        alpha: 1.0,
        beta: 1.0,
        bias: false,
    },
    Cell {
        id: "nt_prism_in_proj",
        op: ResolvedGemmOp::Nt,
        dims: (4621, 384, 1928),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "nn_large_deep",
        op: ResolvedGemmOp::Nn,
        dims: (4096, 3072, 1536),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "tn_large_deep",
        op: ResolvedGemmOp::Tn,
        dims: (4096, 3072, 1536),
        alpha: 1.0,
        beta: 1.0,
        bias: false,
    },
    Cell {
        id: "nt_large_deep",
        op: ResolvedGemmOp::Nt,
        dims: (4096, 3072, 1536),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "nn_large",
        op: ResolvedGemmOp::Nn,
        dims: (2048, 3072, 768),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "tn_large",
        op: ResolvedGemmOp::Tn,
        dims: (2048, 3072, 768),
        alpha: 1.0,
        beta: 1.0,
        bias: false,
    },
    Cell {
        id: "nt_large",
        op: ResolvedGemmOp::Nt,
        dims: (2048, 3072, 768),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "nn_d128_in_proj",
        op: ResolvedGemmOp::Nn,
        dims: (1024, 128, 512),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "tn_d128_in_proj",
        op: ResolvedGemmOp::Tn,
        dims: (1024, 128, 512),
        alpha: 1.0,
        beta: 1.0,
        bias: false,
    },
    Cell {
        id: "nt_d128_in_proj",
        op: ResolvedGemmOp::Nt,
        dims: (1024, 128, 512),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "nn_d128_out_proj",
        op: ResolvedGemmOp::Nn,
        dims: (1024, 256, 128),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
    Cell {
        id: "tn_d128_out_proj",
        op: ResolvedGemmOp::Tn,
        dims: (1024, 256, 128),
        alpha: 1.0,
        beta: 1.0,
        bias: false,
    },
    Cell {
        id: "nt_d128_out_proj",
        op: ResolvedGemmOp::Nt,
        dims: (1024, 256, 128),
        alpha: 1.0,
        beta: 0.0,
        bias: false,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Path {
    Eager,
    Graph,
}

#[derive(Clone, Copy, Debug)]
struct PairedTiming {
    scalar_us: f64,
    candidate_us: f64,
}

impl PairedTiming {
    fn speedup(self) -> f64 {
        self.scalar_us / self.candidate_us
    }
}

#[derive(Clone, Debug)]
struct OrderSamples {
    pairs: Vec<PairedTiming>,
}

impl OrderSamples {
    fn stats(&self) -> Result<OrderStats, String> {
        let scalar = self
            .pairs
            .iter()
            .map(|sample| sample.scalar_us)
            .collect::<Vec<_>>();
        let candidate = self
            .pairs
            .iter()
            .map(|sample| sample.candidate_us)
            .collect::<Vec<_>>();
        let speedups = self
            .pairs
            .iter()
            .copied()
            .map(PairedTiming::speedup)
            .collect::<Vec<_>>();
        Ok(OrderStats {
            scalar_median_us: percentile(&scalar, 0.50)?,
            candidate_median_us: percentile(&candidate, 0.50)?,
            median_speedup: percentile(&speedups, 0.50)?,
            p05_speedup: percentile(&speedups, 0.05)?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct OrderStats {
    scalar_median_us: f64,
    candidate_median_us: f64,
    median_speedup: f64,
    p05_speedup: f64,
}

#[derive(Clone, Copy, Debug)]
struct PairStats {
    ab: OrderStats,
    ba: OrderStats,
}

impl PairStats {
    fn gate(self) -> Result<Self, String> {
        gate_stats(self.ab.median_speedup, self.ab.p05_speedup)?;
        gate_stats(self.ba.median_speedup, self.ba.p05_speedup)?;
        Ok(self)
    }

    fn conservative(self) -> (f64, f64) {
        (
            self.ab.median_speedup.min(self.ba.median_speedup),
            self.ab.p05_speedup.min(self.ba.p05_speedup),
        )
    }
}

#[derive(Clone, Debug)]
struct Samples {
    ab: OrderSamples,
    ba: OrderSamples,
}

impl Samples {
    fn new(ab: OrderSamples, ba: OrderSamples) -> Self {
        Self { ab, ba }
    }

    fn stats(&self) -> Result<PairStats, String> {
        Ok(PairStats {
            ab: self.ab.stats()?,
            ba: self.ba.stats()?,
        })
    }

    fn gate(&self) -> Result<PairStats, String> {
        self.stats()?.gate()
    }
}

fn gate_stats(median: f64, p05: f64) -> Result<(), String> {
    if median < MIN_MEDIAN_SPEEDUP || p05 <= MIN_P05_SPEEDUP {
        return Err(format!(
            "speed gate failed: median={median:.6}, p05={p05:.6}"
        ));
    }
    Ok(())
}

fn request(cell: Cell, route: PhysicalQualificationRoute) -> PhysicalQualificationRequest {
    PhysicalQualificationRequest::contiguous_f32(cell.op, cell.dims, route, cell.epilogue())
}

fn configure(device: &GpuDevice, policy: F32TriadPolicy) -> Result<GpuCtx, String> {
    let ctx = GpuCtx::new(device)?;
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_fast_gemm(false);
    ctx.set_f32_triad_policy(policy);
    Ok(ctx)
}

fn qualify<'a>(
    ctx: &'a GpuCtx,
    request: PhysicalQualificationRequest,
) -> Result<QualifiedPhysicalLaunch<'a>, String> {
    presize_physical_qualification_suite(ctx, &[request])?;
    qualify_physical_launch(ctx, request)
}

fn measure(
    launch: &mut QualifiedPhysicalLaunch<'_>,
    ctx: &GpuCtx,
    path: Path,
    iterations: usize,
) -> Result<f64, String> {
    if iterations == 0 {
        return Err("selector timing iterations must be positive".into());
    }
    let ms = match path {
        Path::Eager => launch.measure_eager_window_ms(ctx, iterations)?,
        Path::Graph => launch.measure_graph_window_ms(ctx, iterations)?,
    };
    let us = ms * 1000.0 / iterations as f64;
    if !us.is_finite() || us <= 0.0 {
        return Err(format!("invalid timing sample {us}"));
    }
    Ok(us)
}

fn calibrate(
    launch: &mut QualifiedPhysicalLaunch<'_>,
    ctx: &GpuCtx,
    path: Path,
) -> Result<usize, String> {
    let per_call_us = measure(launch, ctx, path, CALIBRATION_PROBE_LAUNCHES)?;
    Ok(((TARGET_WINDOW_MS * 1000.0 / per_call_us).ceil() as usize).clamp(1, MAX_WINDOW_LAUNCHES))
}

fn paired_samples(
    candidate: &mut QualifiedPhysicalLaunch<'_>,
    candidate_ctx: &GpuCtx,
    scalar: &mut QualifiedPhysicalLaunch<'_>,
    scalar_ctx: &GpuCtx,
    path: Path,
    windows: usize,
    iterations: usize,
) -> Result<Samples, String> {
    let mut ab = Vec::with_capacity(windows);
    let mut ba = Vec::with_capacity(windows);
    candidate.seed_f32_operands(candidate_ctx, CORPUS_SALT)?;
    scalar.seed_f32_operands(scalar_ctx, CORPUS_SALT)?;
    for window in 0..windows {
        let candidate_ab = measure(candidate, candidate_ctx, path, iterations)?;
        let scalar_ab = measure(scalar, scalar_ctx, path, iterations)?;
        ab.push(PairedTiming {
            scalar_us: scalar_ab,
            candidate_us: candidate_ab,
        });

        let scalar_ba = measure(scalar, scalar_ctx, path, iterations)?;
        let candidate_ba = measure(candidate, candidate_ctx, path, iterations)?;
        let pair = PairedTiming {
            scalar_us: scalar_ba,
            candidate_us: candidate_ba,
        };
        let speedup = pair.speedup();
        if !speedup.is_finite() || speedup <= 0.0 {
            return Err(format!("invalid paired speedup in window {window}"));
        }
        ba.push(pair);
    }
    Ok(Samples::new(
        OrderSamples { pairs: ab },
        OrderSamples { pairs: ba },
    ))
}

fn percentile(values: &[f64], fraction: f64) -> Result<f64, String> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || !(0.0..=1.0).contains(&fraction)
    {
        return Err("percentile requires positive finite samples and a unit fraction".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Ok(sorted[index])
}

fn bit_gate(
    launch: &mut QualifiedPhysicalLaunch<'_>,
    ctx: &GpuCtx,
    reference_bits: &[u32],
    exact_reference: bool,
) -> Result<(), String> {
    let eager = eager_graph_bits(launch, ctx)?;
    if exact_reference {
        if eager != reference_bits {
            return Err("candidate differs bitwise from forced portable TF32 reference".into());
        }
    } else {
        check_numeric_accuracy(&eager, reference_bits)?;
    }
    Ok(())
}

fn check_numeric_accuracy(actual: &[u32], reference: &[u32]) -> Result<(), String> {
    if actual.len() != reference.len() {
        return Err(format!(
            "candidate output length {} differs from reference {}",
            actual.len(),
            reference.len()
        ));
    }
    for (index, (&actual, &reference)) in actual.iter().zip(reference).enumerate() {
        let actual = f64::from(f32::from_bits(actual));
        let reference = f64::from(f32::from_bits(reference));
        let tolerance = 0.0025 * (1.0 + reference.abs());
        if !actual.is_finite() || !reference.is_finite() || (actual - reference).abs() > tolerance {
            return Err(format!(
                "candidate output {index} was {actual}, portable reference {reference}, tolerance {tolerance}"
            ));
        }
    }
    Ok(())
}

fn eager_graph_bits(
    launch: &mut QualifiedPhysicalLaunch<'_>,
    ctx: &GpuCtx,
) -> Result<Vec<u32>, String> {
    launch.seed_f32_operands(ctx, CORPUS_SALT)?;
    measure(launch, ctx, Path::Eager, 1)?;
    let eager = launch.f32_output_bits(ctx)?;
    launch.seed_f32_operands(ctx, CORPUS_SALT)?;
    measure(launch, ctx, Path::Eager, 1)?;
    if eager != launch.f32_output_bits(ctx)? {
        return Err("candidate eager repeat changed output bits".into());
    }
    launch.seed_f32_operands(ctx, CORPUS_SALT)?;
    measure(launch, ctx, Path::Graph, 1)?;
    let graph = launch.f32_output_bits(ctx)?;
    launch.seed_f32_operands(ctx, CORPUS_SALT)?;
    measure(launch, ctx, Path::Graph, 1)?;
    if graph != launch.f32_output_bits(ctx)? {
        return Err("candidate graph repeat changed output bits".into());
    }
    if eager != graph {
        let mismatch = eager
            .iter()
            .zip(&graph)
            .position(|(left, right)| left != right);
        return Err(format!(
            "candidate eager and graph outputs differ bitwise: len={}/{}, first={mismatch:?}, eager={:?}, graph={:?}",
            eager.len(),
            graph.len(),
            mismatch.map(|index| eager[index]),
            mismatch.map(|index| graph[index]),
        ));
    }
    launch.validate_red_zones(ctx)?;
    Ok(eager)
}

fn require_route(
    launch: &QualifiedPhysicalLaunch<'_>,
    module: ModuleKind,
    symbol: Option<&str>,
) -> Result<(), String> {
    let evidence = launch.evidence();
    if !evidence.eager_graph_equal() || evidence.uniform_module_kind() != Some(module) {
        return Err(format!(
            "physical route owner mismatch: {:?}",
            evidence.nodes()
        ));
    }
    if symbol.is_some_and(|expected| !evidence.nodes().iter().any(|node| node.symbol == expected)) {
        return Err(format!(
            "physical route symbol mismatch: {:?}",
            evidence.nodes()
        ));
    }
    Ok(())
}

/// The exact-policy comparator is whatever the exact policy runs for the cell:
/// the scalar module, or the SM120 exact family once the cell has a measured
/// arm there. Either way no node may be a TF32 kernel.
fn require_exact_route(launch: &QualifiedPhysicalLaunch<'_>) -> Result<(), String> {
    let evidence = launch.evidence();
    let module = evidence.uniform_module_kind();
    let exact_module = matches!(
        module,
        Some(ModuleKind::TriadScalar | ModuleKind::TriadSm120)
    );
    let exact_symbols = evidence.nodes().iter().all(|node| {
        !node.symbol.contains("tf32")
            && (node.module_kind != ModuleKind::TriadSm120 || node.symbol.contains("_tma_fma_v1_"))
    });
    if !evidence.eager_graph_equal() || !exact_module || !exact_symbols {
        return Err(format!(
            "exact comparator route mismatch: {:?}",
            evidence.nodes()
        ));
    }
    Ok(())
}

fn portable_reference(ctx: &GpuCtx, cell: Cell) -> Result<Vec<u32>, String> {
    let spec = tf32_route_specs(ModuleKind::TriadSm80)
        .iter()
        .find(|spec| spec.op == cell.op)
        .ok_or_else(|| format!("no portable TF32 reference for {}", cell.id))?;
    let request = request(cell, PhysicalQualificationRoute::Tf32Forced(spec.route));
    let mut launch = qualify(ctx, request)?;
    require_route(&launch, ModuleKind::TriadSm80, Some(spec.symbol))?;
    eager_graph_bits(&mut launch, ctx).map_err(|error| {
        format!(
            "{} portable reference {} bit gate: {error}",
            cell.id, spec.symbol
        )
    })
}

#[derive(Clone, Debug)]
struct CandidateResult {
    symbol: &'static str,
    numeric_family: &'static str,
    request_digest: String,
    launch_digest: String,
    route_identity_json: String,
    eager_iterations: usize,
    graph_iterations: usize,
    eager_calibration_preflight: String,
    eager_timed_preflight: String,
    eager_postflight: String,
    graph_calibration_preflight: String,
    graph_timed_preflight: String,
    graph_postflight: String,
    discovery_eager_median: f64,
    discovery_graph_median: f64,
    final_eager_median: f64,
    final_eager_p05: f64,
    final_graph_median: f64,
    final_graph_p05: f64,
    discovery_eager_orders: PairStats,
    discovery_graph_orders: PairStats,
    final_eager_orders: PairStats,
    final_graph_orders: PairStats,
    discovery_eager_samples: Samples,
    discovery_graph_samples: Samples,
    final_eager_samples: Samples,
    final_graph_samples: Samples,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    route: Tf32PhysicalRoute,
    symbol: &'static str,
    module: ModuleKind,
}

impl Candidate {
    const fn is_split_k(self) -> bool {
        matches!(
            self.route,
            Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_)
                | Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_)
                | Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_)
        )
    }

    /// Stream-K deals the reduction across CTAs and folds it in a fixed order:
    /// deterministic and replay-stable, but its own family of bits.
    const fn is_stream_k(self) -> bool {
        matches!(
            self.route,
            Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_)
        )
    }

    /// Whether the candidate must reproduce the portable reference bit for bit;
    /// the split families are held to the accuracy gate instead.
    const fn shares_reference_bits(self) -> bool {
        !self.is_split_k() && !self.is_stream_k()
    }

    const fn numeric_family(self) -> &'static str {
        if self.is_split_k() {
            "tf32_split_k_fixed_reducer"
        } else if self.is_stream_k() {
            "tf32_sm120_stream_k_fixed_order"
        } else if matches!(self.module, ModuleKind::TriadSm120) {
            "tf32_sm120_tma_direct"
        } else {
            "tf32_direct_portable_association"
        }
    }

    const fn partitions(self) -> Option<usize> {
        match self.route {
            Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_) => Some(2),
            Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_) => Some(4),
            Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_) => Some(8),
            _ => None,
        }
    }
}

fn split_candidates(op: ResolvedGemmOp) -> Vec<Candidate> {
    let route = |tile, stages| Tf32PortableRoute { tile, stages };
    match op {
        ResolvedGemmOp::Nn => vec![
            Candidate {
                route: Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(route(
                    Tf32PortableTile::M16N32,
                    Tf32PortableStages::S4,
                )),
                symbol: "gemm_bi_nn_sm80_mma_tf32_splitk2_v1_m16n32_bk32_s4",
                module: ModuleKind::TriadSm80,
            },
            Candidate {
                route: Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(route(
                    Tf32PortableTile::M16N32,
                    Tf32PortableStages::S4,
                )),
                symbol: "gemm_bi_nn_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4",
                module: ModuleKind::TriadSm80,
            },
        ],
        ResolvedGemmOp::Nt => vec![
            Candidate {
                route: Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(route(
                    Tf32PortableTile::M16N32,
                    Tf32PortableStages::S3,
                )),
                symbol: "gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s3",
                module: ModuleKind::TriadSm80,
            },
            Candidate {
                route: Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(route(
                    Tf32PortableTile::M16N32,
                    Tf32PortableStages::S4,
                )),
                symbol: "gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4",
                module: ModuleKind::TriadSm80,
            },
            Candidate {
                route: Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(route(
                    Tf32PortableTile::M32N32,
                    Tf32PortableStages::S3,
                )),
                symbol: "gemm_bi_nt_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s3",
                module: ModuleKind::TriadSm80,
            },
            Candidate {
                route: Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(route(
                    Tf32PortableTile::M32N32,
                    Tf32PortableStages::S4,
                )),
                symbol: "gemm_bi_nt_sm80_mma_tf32_splitk8_v1_m32n32_bk32_s4",
                module: ModuleKind::TriadSm80,
            },
        ],
        ResolvedGemmOp::Tn => Vec::new(),
    }
}

fn candidates(op: ResolvedGemmOp) -> Vec<Candidate> {
    let mut candidates = tf32_route_specs(ModuleKind::TriadSm80)
        .iter()
        .filter(|spec| spec.op == op)
        .map(|spec| Candidate {
            route: spec.route,
            symbol: spec.symbol,
            module: ModuleKind::TriadSm80,
        })
        .collect::<Vec<_>>();
    candidates.extend(split_candidates(op));
    // The exact-F32 SM120 routes share the module but not the TF32 numeric
    // contract; the exact-family harness qualifies them, not this selector.
    candidates.extend(
        tf32_route_specs(ModuleKind::TriadSm120)
            .iter()
            .filter(|spec| spec.op == op && !spec.route.is_exact_fma())
            .map(|spec| Candidate {
                route: spec.route,
                symbol: spec.symbol,
                module: ModuleKind::TriadSm120,
            }),
    );
    candidates
}

#[derive(Clone, Copy, Debug)]
struct CandidateExclusion {
    symbol: &'static str,
    reason: &'static str,
}

fn tma_strides_are_representable(cell: Cell) -> bool {
    let (lda, ldb) = match cell.op {
        ResolvedGemmOp::Nn | ResolvedGemmOp::Tn => (cell.dims.1, cell.dims.2),
        ResolvedGemmOp::Nt => (cell.dims.2, cell.dims.2),
    };
    lda.is_multiple_of(4) && ldb.is_multiple_of(4)
}

/// The candidates a board can run for `cell`. A board without the SM120
/// module (every board outside that family) still qualifies its portable
/// routes; the SM120 routes are excluded there rather than failing the run.
fn candidate_inventory(
    cell: Cell,
    specialized_bound: bool,
) -> (Vec<Candidate>, Vec<CandidateExclusion>) {
    let output_elements = match cell.op {
        ResolvedGemmOp::Nn => cell.dims.0.checked_mul(cell.dims.2),
        ResolvedGemmOp::Tn => cell.dims.1.checked_mul(cell.dims.2),
        ResolvedGemmOp::Nt => cell.dims.0.checked_mul(cell.dims.1),
    };
    let mut admitted = Vec::new();
    let mut excluded = Vec::new();
    for candidate in candidates(cell.op) {
        let splitk_fits = candidate.partitions().is_none_or(|partitions| {
            output_elements
                .and_then(|elements| elements.checked_mul(partitions))
                .is_some_and(|elements| elements <= FIXED_SPLITK_SCRATCH_ELEMENTS)
        });
        let tma_fits =
            candidate.module != ModuleKind::TriadSm120 || tma_strides_are_representable(cell);
        let module_bound = candidate.module != ModuleKind::TriadSm120 || specialized_bound;
        if !module_bound {
            excluded.push(CandidateExclusion {
                symbol: candidate.symbol,
                reason: "specialized_module_unbound",
            });
        } else if !splitk_fits {
            excluded.push(CandidateExclusion {
                symbol: candidate.symbol,
                reason: "splitk_fixed_workspace",
            });
        } else if !tma_fits {
            excluded.push(CandidateExclusion {
                symbol: candidate.symbol,
                reason: "tma_outer_stride_alignment",
            });
        } else {
            admitted.push(candidate);
        }
    }
    (admitted, excluded)
}

fn candidates_for_cell(cell: Cell) -> Vec<Candidate> {
    candidate_inventory(cell, true).0
}

#[derive(Clone, Debug)]
struct CellResult {
    cell: Cell,
    specialized_identity_json: String,
    candidates: Vec<CandidateResult>,
    excluded_candidates: Vec<CandidateExclusion>,
    discovery_winners: [&'static str; 2],
    final_winners: [&'static str; 2],
    chosen_selection: &'static str,
    qualified: bool,
    admitted: bool,
}

impl CellResult {
    fn json(&self) -> String {
        let candidates = self
            .candidates
            .iter()
            .map(|candidate| {
                format!(
                    concat!(
                        "{{\"symbol\":\"{}\",\"numeric_family\":\"{}\",",
                        "\"request_digest\":\"{}\",\"launch_digest\":\"{}\",",
                        "\"discovery\":{{\"eager_median_speedup\":{},",
                        "\"graph_median_speedup\":{}}},",
                        "\"final\":{{\"eager_median_speedup\":{},",
                        "\"eager_p05_speedup\":{},\"graph_median_speedup\":{},",
                        "\"graph_p05_speedup\":{}}},",
                        "\"iterations\":{{\"eager\":{},\"graph\":{}}},",
                        "\"qualification_identity\":{},",
                        "\"quiet\":{{\"eager\":{{\"calibration\":\"{}\",",
                        "\"timed\":\"{}\",\"post\":\"{}\"}},",
                        "\"graph\":{{\"calibration\":\"{}\",",
                        "\"timed\":\"{}\",\"post\":\"{}\"}}}},",
                        "\"order_stats\":{{\"discovery_eager\":{},",
                        "\"discovery_graph\":{},\"final_eager\":{},",
                        "\"final_graph\":{}}},",
                        "\"raw_samples\":{{\"discovery_eager\":{},",
                        "\"discovery_graph\":{},\"final_eager\":{},",
                        "\"final_graph\":{}}}}}"
                    ),
                    candidate.symbol,
                    candidate.numeric_family,
                    candidate.request_digest,
                    candidate.launch_digest,
                    candidate.discovery_eager_median,
                    candidate.discovery_graph_median,
                    candidate.final_eager_median,
                    candidate.final_eager_p05,
                    candidate.final_graph_median,
                    candidate.final_graph_p05,
                    candidate.eager_iterations,
                    candidate.graph_iterations,
                    candidate.route_identity_json,
                    escape_json_string(&candidate.eager_calibration_preflight),
                    escape_json_string(&candidate.eager_timed_preflight),
                    escape_json_string(&candidate.eager_postflight),
                    escape_json_string(&candidate.graph_calibration_preflight),
                    escape_json_string(&candidate.graph_timed_preflight),
                    escape_json_string(&candidate.graph_postflight),
                    pair_stats_json(candidate.discovery_eager_orders),
                    pair_stats_json(candidate.discovery_graph_orders),
                    pair_stats_json(candidate.final_eager_orders),
                    pair_stats_json(candidate.final_graph_orders),
                    pair_samples_json(&candidate.discovery_eager_samples),
                    pair_samples_json(&candidate.discovery_graph_samples),
                    pair_samples_json(&candidate.final_eager_samples),
                    pair_samples_json(&candidate.final_graph_samples),
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let excluded = self
            .excluded_candidates
            .iter()
            .map(|candidate| {
                format!(
                    "{{\"symbol\":\"{}\",\"reason\":\"{}\"}}",
                    candidate.symbol, candidate.reason
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{\"schema\":\"{}\",\"cell_id\":\"{}\",\"op\":\"{}\",",
                "\"dims\":[{},{},{}],",
                "\"epilogue\":{{\"alpha\":{},\"beta\":{},\"bias\":{}}},",
                "\"specialized_identity\":{},",
                "\"candidates\":[{}],",
                "\"excluded_candidates\":[{}],",
                "\"discovery_winners\":{{\"eager\":\"{}\",\"graph\":\"{}\"}},",
                "\"final_winners\":{{\"eager\":\"{}\",\"graph\":\"{}\"}},",
                "\"chosen_selection\":\"{}\",",
                "\"qualified\":{},\"admitted\":{}}}"
            ),
            SCHEMA,
            self.cell.id,
            op_name(self.cell.op),
            self.cell.dims.0,
            self.cell.dims.1,
            self.cell.dims.2,
            self.cell.alpha,
            self.cell.beta,
            self.cell.bias,
            self.specialized_identity_json,
            candidates,
            excluded,
            self.discovery_winners[0],
            self.discovery_winners[1],
            self.final_winners[0],
            self.final_winners[1],
            self.chosen_selection,
            self.qualified,
            self.admitted,
        )
    }
}

fn pair_stats_json(stats: PairStats) -> String {
    let order = |stats: OrderStats| {
        format!(
            concat!(
                "{{\"scalar_median_us\":{},\"candidate_median_us\":{},",
                "\"median_speedup\":{},\"p05_speedup\":{}}}"
            ),
            stats.scalar_median_us,
            stats.candidate_median_us,
            stats.median_speedup,
            stats.p05_speedup,
        )
    };
    format!("{{\"ab\":{},\"ba\":{}}}", order(stats.ab), order(stats.ba))
}

fn pair_samples_json(samples: &Samples) -> String {
    let order = |samples: &OrderSamples| {
        let pairs = samples
            .pairs
            .iter()
            .map(|sample| {
                format!(
                    "{{\"scalar_us\":{},\"candidate_us\":{},\"speedup\":{}}}",
                    sample.scalar_us,
                    sample.candidate_us,
                    sample.speedup(),
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("[{pairs}]")
    };
    format!(
        "{{\"ab\":{},\"ba\":{}}}",
        order(&samples.ab),
        order(&samples.ba)
    )
}

fn route_identity_json(identity: GemmRouteIdentity) -> String {
    format!(
        concat!(
            "{{\"compiler_target\":\"{}\",\"nvrtc_version\":[{},{}],",
            "\"nvrtc_library_known\":{},\"source_digest\":\"{}\",",
            "\"invocation_digest\":\"{}\",\"header_manifest_digest\":\"{}\",",
            "\"nvrtc_library_domain\":\"{}\",\"artifact_set_digest\":\"{}\",",
            "\"device_cc\":[{},{}],\"multiprocessor_count\":{},",
            "\"device_target\":\"{}\",\"driver_api_version\":{},",
            "\"driver_build_digest\":\"{}\",\"policy_hash\":\"{}\",",
            "\"tuning_table_revision\":{},\"schedule_set_revision\":{}}}"
        ),
        identity.compiler.target.as_str(),
        identity.compiler.nvrtc_version.0,
        identity.compiler.nvrtc_version.1,
        identity.compiler.nvrtc_library_known,
        digest_hex(&identity.compiler.source_digest),
        digest_hex(&identity.compiler.invocation_digest),
        digest_hex(&identity.compiler.header_manifest_digest),
        digest_hex(&identity.compiler.nvrtc_library_domain),
        digest_hex(&identity.artifacts.ordered_digest),
        identity.device.compute_capability.0,
        identity.device.compute_capability.1,
        identity.device.multiprocessor_count,
        identity.device.target.as_str(),
        identity.device.driver.api_version,
        digest_hex(&identity.device.driver.build_digest),
        digest_hex(&identity.policy_hash),
        identity.tuning_table_revision,
        identity.schedule_set_revision,
    )
}

fn qualified_module_json(module: Tf32QualifiedModule) -> String {
    let accepted_target = module.device_caps.accepted_target.map_or_else(
        || "null".to_string(),
        |target| format!("\"{}\"", target.as_str()),
    );
    format!(
        concat!(
            "{{\"module_kind\":\"{:?}\",\"module_target\":\"{}\",",
            "\"artifact_module_kind\":\"{:?}\",\"artifact_kind\":\"{:?}\",",
            "\"compile_key\":\"{}\",\"artifact_digest\":\"{}\",",
            "\"compiler_target\":\"{}\",\"nvrtc_version\":[{},{}],",
            "\"nvrtc_library_known\":{},\"source_digest\":\"{}\",",
            "\"invocation_digest\":\"{}\",\"header_manifest_digest\":\"{}\",",
            "\"nvrtc_library_domain\":\"{}\",\"output_kind\":\"{:?}\",",
            "\"composer_revision\":{},\"compiler_revision\":{},",
            "\"numeric_abi_revision\":{},\"schedule_revision\":{},",
            "\"device_cc\":[{},{}],\"multiprocessor_count\":{},",
            "\"device_target\":\"{}\",\"driver_api_version\":{},",
            "\"driver_build_sources\":{},\"driver_build_digest\":\"{}\",",
            "\"caps_cc\":[{},{}],\"caps_nvrtc_version\":[{},{}],",
            "\"caps_accepted_target\":{},\"caps_optin_shared_bytes\":{},",
            "\"caps_tensor_map_access\":{}}}"
        ),
        module.module_kind,
        module.target.as_str(),
        module.artifact.module_kind,
        module.artifact.artifact_kind,
        digest_hex(&module.artifact.compile_key),
        digest_hex(&module.artifact.artifact_digest),
        module.compiler.target.as_str(),
        module.compiler.nvrtc_version.0,
        module.compiler.nvrtc_version.1,
        module.compiler.nvrtc_library_known,
        digest_hex(&module.compiler.source_digest),
        digest_hex(&module.compiler.invocation_digest),
        digest_hex(&module.compiler.header_manifest_digest),
        digest_hex(&module.compiler.nvrtc_library_domain),
        module.compiler.output_kind,
        module.compiler.composer_revision,
        module.compiler.compiler_revision,
        module.compiler.numeric_abi_revision,
        module.compiler.schedule_revision,
        module.device.compute_capability.0,
        module.device.compute_capability.1,
        module.device.multiprocessor_count,
        module.device.target.as_str(),
        module.device.driver.api_version,
        module.device.driver.build_sources,
        digest_hex(&module.device.driver.build_digest),
        module.device_caps.compute_capability.0,
        module.device_caps.compute_capability.1,
        module.device_caps.nvrtc_version.0,
        module.device_caps.nvrtc_version.1,
        accepted_target,
        module.device_caps.optin_shared_bytes,
        module.device_caps.tensor_map_access,
    )
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
                escaped.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => escaped.push(character),
        }
    }
    escaped
}

fn json_delimiters_are_balanced(value: &str) -> bool {
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for byte in value.bytes() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => stack.push(byte),
            b'}' if stack.pop() != Some(b'{') => return false,
            b']' if stack.pop() != Some(b'[') => return false,
            _ => {}
        }
    }
    !in_string && !escaped && stack.is_empty()
}

fn select_winner(
    candidates: &[Candidate],
    stats: &BTreeMap<(Path, &'static str), (f64, f64)>,
    path: Path,
) -> &'static str {
    candidates
        .iter()
        .filter(|candidate| {
            let (median, p05) = stats[&(path, candidate.symbol)];
            median >= MIN_MEDIAN_SPEEDUP && p05 > MIN_P05_SPEEDUP
        })
        .max_by(|left, right| {
            stats[&(path, left.symbol)]
                .0
                .total_cmp(&stats[&(path, right.symbol)].0)
        })
        .map_or("scalar_fma_v1", |candidate| candidate.symbol)
}

fn stable_candidate_score(
    candidate: Candidate,
    discovery: &BTreeMap<(Path, &'static str), (f64, f64)>,
    final_stats: &BTreeMap<(Path, &'static str), (f64, f64)>,
) -> Option<(f64, f64)> {
    let mut median_floor = f64::INFINITY;
    let mut p05_floor = f64::INFINITY;
    for stats in [discovery, final_stats] {
        for path in [Path::Eager, Path::Graph] {
            let (median, p05) = *stats.get(&(path, candidate.symbol))?;
            gate_stats(median, p05).ok()?;
            median_floor = median_floor.min(median);
            p05_floor = p05_floor.min(p05);
        }
    }
    Some((median_floor, p05_floor))
}

fn select_stable_winner(
    candidates: &[Candidate],
    discovery: &BTreeMap<(Path, &'static str), (f64, f64)>,
    final_stats: &BTreeMap<(Path, &'static str), (f64, f64)>,
) -> &'static str {
    candidates
        .iter()
        .copied()
        .filter_map(|candidate| {
            stable_candidate_score(candidate, discovery, final_stats)
                .map(|score| (candidate, score))
        })
        .max_by(|(left, left_score), (right, right_score)| {
            left_score
                .0
                .total_cmp(&right_score.0)
                .then_with(|| left_score.1.total_cmp(&right_score.1))
                .then_with(|| right.symbol.cmp(left.symbol))
        })
        .map_or("scalar_fma_v1", |(candidate, _)| candidate.symbol)
}

struct JsonlSink {
    path: PathBuf,
    writer: BufWriter<File>,
    records: usize,
    digest: Sha256,
}

impl JsonlSink {
    fn create_from_env() -> Result<Self, String> {
        let path = std::env::var_os(OUTPUT_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .ok_or_else(|| format!("{OUTPUT_ENV} must name a new JSONL file"))?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| format!("create SM120 selector artifact {path:?}: {error}"))?;
        if !file
            .metadata()
            .map_err(|error| format!("inspect SM120 selector artifact {path:?}: {error}"))?
            .is_file()
        {
            return Err(format!(
                "SM120 selector artifact {path:?} is not a regular file"
            ));
        }
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            records: 0,
            digest: Sha256::new(),
        })
    }

    fn write(&mut self, result: &CellResult) -> Result<(), String> {
        let record = result.json();
        writeln!(self.writer, "{record}")
            .map_err(|error| format!("write SM120 selector artifact {:?}: {error}", self.path))?;
        self.digest.update(record.as_bytes());
        self.digest.update(b"\n");
        self.records += 1;
        Ok(())
    }

    fn finish(
        mut self,
        device: &GpuDevice,
        expected_records: usize,
        pre_context: &str,
        postflight: &str,
    ) -> Result<(), String> {
        if self.records != expected_records {
            return Err(format!(
                "SM120 selector artifact has {} records, expected {}",
                self.records, expected_records
            ));
        }
        let digest: [u8; 32] = self.digest.finalize().into();
        writeln!(
            self.writer,
            "{}",
            completion_record(device, self.records, digest, pre_context, postflight)
        )
        .map_err(|error| format!("write SM120 selector completion {:?}: {error}", self.path))?;
        self.writer
            .flush()
            .map_err(|error| format!("flush SM120 selector artifact {:?}: {error}", self.path))?;
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("sync SM120 selector artifact {:?}: {error}", self.path))
    }
}

fn completion_record(
    device: &GpuDevice,
    records: usize,
    records_digest: [u8; 32],
    pre_context: &str,
    postflight: &str,
) -> String {
    let identity = device.identity();
    format!(
        concat!(
            "{{\"schema\":\"{}\",\"record_type\":\"completion\",",
            "\"cell_records\":{},\"cell_records_sha256\":\"{}\",",
            "\"discovery_windows_per_order\":{},\"final_windows_per_order\":{},",
            "\"target_window_ms\":{},\"warmup_launches_per_arm\":{},",
            "\"device_cc\":[{},{}],\"multiprocessor_count\":{},",
            "\"device_target\":\"{}\",\"nvrtc_target\":\"{}\",",
            "\"driver_api_version\":{},\"driver_build_digest\":\"{}\",",
            "\"pre_context\":\"{}\",\"postflight\":\"{}\",\"passed\":true}}"
        ),
        SCHEMA,
        records,
        digest_hex(&records_digest),
        DISCOVERY_WINDOWS,
        FINAL_WINDOWS,
        TARGET_WINDOW_MS,
        WARMUP_LAUNCHES,
        identity.compute_capability.0,
        identity.compute_capability.1,
        identity.multiprocessor_count,
        identity.target.as_str(),
        device.nvrtc_target(),
        identity.driver.api_version,
        digest_hex(&identity.driver.build_digest),
        escape_json_string(pre_context),
        escape_json_string(postflight),
    )
}

fn op_name(op: ResolvedGemmOp) -> &'static str {
    match op {
        ResolvedGemmOp::Nn => "nn",
        ResolvedGemmOp::Tn => "tn",
        ResolvedGemmOp::Nt => "nt",
    }
}

fn run_cell(device: &GpuDevice, cell: Cell, quiet: &QuietGpu) -> Result<CellResult, String> {
    let candidate_ctx = configure(device, F32TriadPolicy::AllowDeterministicTf32V1)?;
    let scalar_ctx = configure(device, F32TriadPolicy::ExactScalarFmaV1)?;
    // The identity the cohort is frozen against: the SM120 module where the
    // board has one, otherwise the portable SM80 module the board runs.
    let availability = candidate_ctx.kernels.f32_triad_availability();
    let specialized_bound = availability.specialized.is_some();
    let specialized_identity_json = availability
        .specialized
        .or(availability.portable)
        .map(qualified_module_json)
        .ok_or_else(|| {
            "selector qualification has no TF32 module bound on this board".to_string()
        })?;
    let reference_bits = portable_reference(&candidate_ctx, cell)?;
    let mut scalar_gate = qualify(
        &scalar_ctx,
        request(
            cell,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
        ),
    )?;
    require_exact_route(&scalar_gate)?;
    let scalar_nodes = format!("{:?}", scalar_gate.evidence().nodes());
    let scalar_reference_bits =
        eager_graph_bits(&mut scalar_gate, &scalar_ctx).map_err(|error| {
            format!(
                "{} scalar reference bit gate: {error}; nodes={scalar_nodes}",
                cell.id
            )
        })?;
    let (candidates, excluded_candidates) = candidate_inventory(cell, specialized_bound);

    let mut discovery = BTreeMap::new();
    let mut final_stats = BTreeMap::new();
    let mut discovery_orders = BTreeMap::new();
    let mut final_orders = BTreeMap::new();
    let mut discovery_raw = BTreeMap::new();
    let mut final_raw = BTreeMap::new();
    let mut identities = BTreeMap::new();
    let mut iterations = BTreeMap::new();
    let mut calibration_preflights = BTreeMap::new();
    let mut timed_preflights = BTreeMap::new();
    let mut postflights = BTreeMap::new();
    for spec in &candidates {
        let label = format!("sm120-selector/{}/{}", cell.id, spec.symbol);
        let candidate_request = request(cell, PhysicalQualificationRoute::Tf32Forced(spec.route));
        let mut candidate = qualify(&candidate_ctx, candidate_request)?;
        require_route(&candidate, spec.module, Some(spec.symbol))?;
        identities.insert(
            spec.symbol,
            (
                digest_hex(&candidate.evidence().request_identity_digest()),
                digest_hex(&candidate.evidence().launch_digest()),
                route_identity_json(*candidate.evidence().route_identity()),
            ),
        );
        bit_gate(
            &mut candidate,
            &candidate_ctx,
            &reference_bits,
            spec.shares_reference_bits(),
        )
        .map_err(|error| format!("{} {} bit gate: {error}", cell.id, spec.symbol))?;
        for path in [Path::Eager, Path::Graph] {
            let path_name = match path {
                Path::Eager => "eager",
                Path::Graph => "graph",
            };
            let calibration_preflight =
                quiet.require_cohort(&format!("{label}/{path_name}/calibration"))?;
            candidate.seed_f32_operands(&candidate_ctx, CORPUS_SALT)?;
            scalar_gate.seed_f32_operands(&scalar_ctx, CORPUS_SALT)?;
            measure(&mut candidate, &candidate_ctx, path, WARMUP_LAUNCHES)?;
            measure(&mut scalar_gate, &scalar_ctx, path, WARMUP_LAUNCHES)?;
            let candidate_iterations = calibrate(&mut candidate, &candidate_ctx, path)?;
            let scalar_iterations = calibrate(&mut scalar_gate, &scalar_ctx, path)?;
            let window_iterations = candidate_iterations.max(scalar_iterations);
            let timed_preflight = quiet.require_cohort(&format!("{label}/{path_name}/timed"))?;
            let discovery_samples = paired_samples(
                &mut candidate,
                &candidate_ctx,
                &mut scalar_gate,
                &scalar_ctx,
                path,
                DISCOVERY_WINDOWS,
                window_iterations,
            )?;
            let discovery_pair = discovery_samples.stats()?;
            discovery.insert((path, spec.symbol), discovery_pair.conservative());
            discovery_orders.insert((path, spec.symbol), discovery_pair);
            discovery_raw.insert((path, spec.symbol), discovery_samples);
            let final_samples = paired_samples(
                &mut candidate,
                &candidate_ctx,
                &mut scalar_gate,
                &scalar_ctx,
                path,
                FINAL_WINDOWS,
                window_iterations,
            )?;
            let final_pair = final_samples.stats()?;
            final_stats.insert((path, spec.symbol), final_pair.conservative());
            final_orders.insert((path, spec.symbol), final_pair);
            final_raw.insert((path, spec.symbol), final_samples);
            let postflight = quiet.verify_post_cohort(&format!("{label}/{path_name}/post"))?;
            iterations.insert((path, spec.symbol), window_iterations);
            calibration_preflights.insert((path, spec.symbol), calibration_preflight);
            timed_preflights.insert((path, spec.symbol), timed_preflight);
            postflights.insert((path, spec.symbol), postflight);
        }
        candidate.validate_red_zones(&candidate_ctx)?;
        scalar_gate.validate_red_zones(&scalar_ctx)?;
        bit_gate(
            &mut candidate,
            &candidate_ctx,
            &reference_bits,
            spec.shares_reference_bits(),
        )
        .map_err(|error| format!("{} {} final bit gate: {error}", cell.id, spec.symbol))?;
        let scalar_final = eager_graph_bits(&mut scalar_gate, &scalar_ctx)?;
        if scalar_final != scalar_reference_bits {
            return Err(format!(
                "{} scalar comparator changed output bits after timing {}",
                cell.id, spec.symbol
            ));
        }
    }
    let winners =
        [Path::Eager, Path::Graph].map(|path| select_winner(&candidates, &discovery, path));
    let final_winners =
        [Path::Eager, Path::Graph].map(|path| select_winner(&candidates, &final_stats, path));
    let chosen_selection = select_stable_winner(&candidates, &discovery, &final_stats);
    if chosen_selection != "scalar_fma_v1" {
        for path in [Path::Eager, Path::Graph] {
            let (median, p05) = final_stats[&(path, chosen_selection)];
            gate_stats(median, p05).map_err(|error| {
                format!(
                    "{} {path:?} winner {}: {error}; discovery={discovery:?}; final={final_stats:?}",
                    cell.id, chosen_selection
                )
            })?;
        }
    }
    let candidate_results = candidates
        .iter()
        .map(|spec| CandidateResult {
            symbol: spec.symbol,
            numeric_family: spec.numeric_family(),
            request_digest: identities[spec.symbol].0.clone(),
            launch_digest: identities[spec.symbol].1.clone(),
            route_identity_json: identities[spec.symbol].2.clone(),
            eager_iterations: iterations[&(Path::Eager, spec.symbol)],
            graph_iterations: iterations[&(Path::Graph, spec.symbol)],
            eager_calibration_preflight: calibration_preflights[&(Path::Eager, spec.symbol)]
                .clone(),
            eager_timed_preflight: timed_preflights[&(Path::Eager, spec.symbol)].clone(),
            eager_postflight: postflights[&(Path::Eager, spec.symbol)].clone(),
            graph_calibration_preflight: calibration_preflights[&(Path::Graph, spec.symbol)]
                .clone(),
            graph_timed_preflight: timed_preflights[&(Path::Graph, spec.symbol)].clone(),
            graph_postflight: postflights[&(Path::Graph, spec.symbol)].clone(),
            discovery_eager_median: discovery[&(Path::Eager, spec.symbol)].0,
            discovery_graph_median: discovery[&(Path::Graph, spec.symbol)].0,
            final_eager_median: final_stats[&(Path::Eager, spec.symbol)].0,
            final_eager_p05: final_stats[&(Path::Eager, spec.symbol)].1,
            final_graph_median: final_stats[&(Path::Graph, spec.symbol)].0,
            final_graph_p05: final_stats[&(Path::Graph, spec.symbol)].1,
            discovery_eager_orders: discovery_orders[&(Path::Eager, spec.symbol)],
            discovery_graph_orders: discovery_orders[&(Path::Graph, spec.symbol)],
            final_eager_orders: final_orders[&(Path::Eager, spec.symbol)],
            final_graph_orders: final_orders[&(Path::Graph, spec.symbol)],
            discovery_eager_samples: discovery_raw[&(Path::Eager, spec.symbol)].clone(),
            discovery_graph_samples: discovery_raw[&(Path::Graph, spec.symbol)].clone(),
            final_eager_samples: final_raw[&(Path::Eager, spec.symbol)].clone(),
            final_graph_samples: final_raw[&(Path::Graph, spec.symbol)].clone(),
        })
        .collect();
    Ok(CellResult {
        cell,
        specialized_identity_json,
        candidates: candidate_results,
        excluded_candidates,
        discovery_winners: winners,
        final_winners,
        chosen_selection,
        qualified: true,
        admitted: chosen_selection != "scalar_fma_v1",
    })
}

#[test]
#[ignore = "requires an explicitly enabled idle CC12.0/170SM GPU"]
fn sm120_tf32_selector_qualification() -> Result<(), String> {
    match std::env::var(ENABLE_ENV) {
        Ok(value) if value == "1" => {}
        Err(std::env::VarError::NotPresent) => {
            eprintln!("skipping: set {ENABLE_ENV}=1 to run qualification");
            return Ok(());
        }
        Ok(value) => return Err(format!("{ENABLE_ENV} must equal 1, got {value:?}")),
        Err(error) => return Err(format!("read {ENABLE_ENV}: {error}")),
    }
    if cfg!(debug_assertions) {
        return Err("SM120 TF32 selector qualification requires --release".into());
    }
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let pre_context = quiet.require_pre_context("sm120-selector/pre-context")?;
    let device = GpuDevice::new(0)?;
    // Any board with a bound TF32 module qualifies its own cohort; the JSONL
    // records the board, and the cohort it freezes is scoped to it.
    if configure(&device, F32TriadPolicy::AllowDeterministicTf32V1)?
        .kernels
        .f32_triad_availability()
        .portable
        .is_none()
    {
        eprintln!(
            "skipping: no portable TF32 module is bound on {:?}/{}SM",
            device.compute_capability,
            device.multiprocessor_count()
        );
        return Ok(());
    }
    eprintln!(
        "selector qualification on {:?}/{}SM",
        device.compute_capability,
        device.multiprocessor_count()
    );
    let mut sink = JsonlSink::create_from_env()?;
    for cell in CELLS {
        quiet.require_cohort(&format!("sm120-selector/{}/pre-cell", cell.id))?;
        sink.write(&run_cell(&device, cell, &quiet)?)?;
        quiet.verify_post_cohort(&format!("sm120-selector/{}/post-cell", cell.id))?;
    }
    let postflight = quiet.verify_post_cohort("sm120-selector/post-suite")?;
    sink.finish(&device, CELLS.len(), &pre_context, &postflight)?;
    eprintln!(
        "{{\"schema\":\"{SCHEMA}\",\"cells\":{},\"discovery_windows_per_order\":{DISCOVERY_WINDOWS},\"final_windows_per_order\":{FINAL_WINDOWS},\"passed\":true}}",
        CELLS.len(),
    );
    Ok(())
}

#[test]
#[ignore = "requires an explicitly enabled idle CC12.0/170SM GPU"]
fn sm120_tf32_projection_selector_qualification() -> Result<(), String> {
    match std::env::var(ENABLE_ENV) {
        Ok(value) if value == "1" => {}
        Err(std::env::VarError::NotPresent) => {
            eprintln!("skipping: set {ENABLE_ENV}=1 to run qualification");
            return Ok(());
        }
        Ok(value) => return Err(format!("{ENABLE_ENV} must equal 1, got {value:?}")),
        Err(error) => return Err(format!("read {ENABLE_ENV}: {error}")),
    }
    if cfg!(debug_assertions) {
        return Err("SM120 TF32 selector qualification requires --release".into());
    }
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let pre_context = quiet.require_pre_context("sm120-projection-selector/pre-context")?;
    let device = GpuDevice::new(0)?;
    // Any board with a bound TF32 module qualifies its own cohort; the JSONL
    // records the board, and the cohort it freezes is scoped to it.
    if configure(&device, F32TriadPolicy::AllowDeterministicTf32V1)?
        .kernels
        .f32_triad_availability()
        .portable
        .is_none()
    {
        eprintln!(
            "skipping: no portable TF32 module is bound on {:?}/{}SM",
            device.compute_capability,
            device.multiprocessor_count()
        );
        return Ok(());
    }
    eprintln!(
        "selector qualification on {:?}/{}SM",
        device.compute_capability,
        device.multiprocessor_count()
    );
    let mut sink = JsonlSink::create_from_env()?;
    for cell in PROJECTION_CELLS {
        quiet.require_cohort(&format!("sm120-projection-selector/{}/pre-cell", cell.id))?;
        sink.write(&run_cell(&device, cell, &quiet)?)?;
        quiet.verify_post_cohort(&format!("sm120-projection-selector/{}/post-cell", cell.id))?;
    }
    let postflight = quiet.verify_post_cohort("sm120-projection-selector/post-suite")?;
    sink.finish(&device, PROJECTION_CELLS.len(), &pre_context, &postflight)?;
    eprintln!(
        "{{\"schema\":\"{SCHEMA}\",\"projection_cells\":{},\"discovery_windows_per_order\":{DISCOVERY_WINDOWS},\"final_windows_per_order\":{FINAL_WINDOWS},\"passed\":true}}",
        PROJECTION_CELLS.len(),
    );
    Ok(())
}

#[test]
fn qualification_plan_is_exact() {
    assert_eq!(CELLS.len(), 18);
    assert_eq!(CELLS[0].dims, (64, 384, 1536));
    assert_eq!(
        CELLS[0].epilogue(),
        PhysicalQualificationF32Epilogue::new(1.0, 0.0, true)
    );
    assert_eq!(CELLS[1].dims, (64, 1536, 384));
    assert_eq!(CELLS[2].dims, (64, 833, 384));
    assert_eq!(CELLS[3].dims, (64, 384, 101));
    assert_eq!(CELLS[7].dims, (64, 384, 1536));
    assert_eq!(CELLS[8].dims, (64, 1536, 384));
    for cell in &CELLS[7..9] {
        assert_eq!(
            cell.epilogue(),
            PhysicalQualificationF32Epilogue::new(1.0, 1.0, false)
        );
    }
    assert_eq!(CELLS[14].dims, (10400, 768, 384));
    assert_eq!(CELLS[15].dims, (10400, 384, 1536));
    assert_eq!(CELLS[16].dims, (10400, 768, 384));
    assert_eq!(CELLS[17].dims, (10400, 768, 384));
    assert_eq!(
        CELLS
            .iter()
            .filter(|cell| cell.op == ResolvedGemmOp::Nn)
            .count(),
        11
    );
    assert_eq!(
        CELLS
            .iter()
            .filter(|cell| cell.op == ResolvedGemmOp::Tn)
            .count(),
        4
    );
    assert_eq!(
        CELLS
            .iter()
            .filter(|cell| cell.op == ResolvedGemmOp::Nt)
            .count(),
        3
    );
    let tf32_routes = |op| {
        tf32_route_specs(ModuleKind::TriadSm120)
            .iter()
            .filter(|spec| spec.op == op && !spec.route.is_exact_fma())
            .count()
    };
    assert_eq!(tf32_routes(ResolvedGemmOp::Nn), 6);
    assert_eq!(tf32_routes(ResolvedGemmOp::Tn), 7);
    assert_eq!(tf32_routes(ResolvedGemmOp::Nt), 5);
    assert_eq!(
        tf32_route_specs(ModuleKind::TriadSm120)
            .iter()
            .filter(|spec| spec.route.is_exact_fma())
            .count(),
        12
    );
    assert_eq!(candidates(ResolvedGemmOp::Nn).len(), 14);
    assert_eq!(candidates(ResolvedGemmOp::Tn).len(), 13);
    assert_eq!(candidates(ResolvedGemmOp::Nt).len(), 15);
    assert_eq!(DISCOVERY_WINDOWS, 21);
    assert_eq!(FINAL_WINDOWS, 101);
}

#[test]
fn paired_percentiles_and_speed_gate_are_fail_closed() {
    let values = (1..=202).map(|value| value as f64).collect::<Vec<_>>();
    assert_eq!(percentile(&values, 0.05).unwrap(), 11.0);
    assert_eq!(percentile(&values, 0.50).unwrap(), 101.0);
    let order = |speedup| OrderSamples {
        pairs: vec![
            PairedTiming {
                scalar_us: speedup,
                candidate_us: 1.0,
            };
            101
        ],
    };
    assert!(Samples::new(order(1.02), order(1.01)).gate().is_ok());
    assert!(Samples::new(order(1.02), order(1.0)).gate().is_err());
    assert!(Samples::new(order(0.99), order(1.03)).gate().is_err());
}

#[test]
fn runtime_qualification_requires_gpu_quiet_gates() {
    let start = TEST_SOURCE
        .find("fn sm120_tf32_selector_qualification()")
        .expect("selector runtime entry");
    let implementation = &TEST_SOURCE[..start];
    let tail = &TEST_SOURCE[start..];
    let end = tail
        .find("\n#[test]")
        .expect("test following selector runtime");
    let runtime = &tail[..end];
    for required in [
        "QuietGpu::for_cuda_ordinal",
        "require_pre_context",
        "require_cohort",
        "verify_post_cohort",
        "run_cell",
        "sink.finish",
    ] {
        assert!(
            runtime.contains(required),
            "selector qualification lost quiet gate {required}"
        );
    }
    for required in [
        "const TARGET_WINDOW_MS: f64 = 5.0;",
        "WARMUP_LAUNCHES",
        "calibrate",
        "ms * 1000.0 / iterations as f64",
        "check_numeric_accuracy",
        "record_type",
    ] {
        assert!(
            implementation.contains(required),
            "selector qualification lost suite contract {required}"
        );
    }
}

#[test]
fn selector_returns_scalar_unless_one_candidate_clears_both_gates() {
    let candidates = candidates(ResolvedGemmOp::Nn);
    let mut stats = BTreeMap::new();
    for candidate in &candidates {
        stats.insert((Path::Eager, candidate.symbol), (1.009, 1.008));
    }
    assert_eq!(
        select_winner(&candidates, &stats, Path::Eager),
        "scalar_fma_v1"
    );
    stats.insert((Path::Eager, candidates[3].symbol), (1.02, 1.001));
    assert_eq!(
        select_winner(&candidates, &stats, Path::Eager),
        candidates[3].symbol
    );
}

#[test]
fn stable_selector_uses_the_best_candidate_that_clears_every_cohort() {
    let candidates = candidates(ResolvedGemmOp::Nn);
    let mut discovery = BTreeMap::new();
    let mut final_stats = BTreeMap::new();
    for candidate in &candidates {
        for path in [Path::Eager, Path::Graph] {
            discovery.insert((path, candidate.symbol), (1.0, 0.99));
            final_stats.insert((path, candidate.symbol), (1.0, 0.99));
        }
    }
    for (path, median) in [(Path::Eager, 1.5), (Path::Graph, 1.2)] {
        discovery.insert((path, candidates[0].symbol), (median, 1.001));
    }
    for (path, median) in [(Path::Eager, 1.1), (Path::Graph, 1.09)] {
        final_stats.insert((path, candidates[0].symbol), (median, 1.001));
    }
    for path in [Path::Eager, Path::Graph] {
        discovery.insert((path, candidates[1].symbol), (1.11, 1.002));
        final_stats.insert((path, candidates[1].symbol), (1.4, 1.002));
    }

    assert_eq!(
        select_winner(&candidates, &discovery, Path::Eager),
        candidates[0].symbol
    );
    assert_eq!(
        select_winner(&candidates, &final_stats, Path::Eager),
        candidates[1].symbol
    );
    assert_eq!(
        select_stable_winner(&candidates, &discovery, &final_stats),
        candidates[1].symbol
    );
}

#[test]
fn split_candidate_inventory_is_exact_and_operation_scoped() {
    let nn = split_candidates(ResolvedGemmOp::Nn);
    let tn = split_candidates(ResolvedGemmOp::Tn);
    let nt = split_candidates(ResolvedGemmOp::Nt);
    assert_eq!(nn.len(), 2);
    assert!(tn.is_empty());
    assert_eq!(nt.len(), 4);
    assert!(nn.iter().all(|candidate| candidate.symbol.contains("_nn_")));
    assert!(nt.iter().all(|candidate| candidate.symbol.contains("_nt_")));
    assert!(
        nn.iter()
            .chain(&nt)
            .all(|candidate| candidate.module == ModuleKind::TriadSm80)
    );
}

#[test]
fn split_candidates_that_exceed_the_fixed_workspace_are_not_timed() {
    assert!(DISPATCH_SOURCE.contains("pub(super) const SPLITK_SCRATCH_CAP: usize = 1 << 23;"));
    assert_eq!(candidates_for_cell(CELLS[0]).len(), 14);
    assert_eq!(candidates_for_cell(CELLS[9]).len(), 13);
    assert_eq!(candidates_for_cell(CELLS[14]).len(), 13);
    assert_eq!(candidates_for_cell(CELLS[17]).len(), 11);
    assert_eq!(candidates_for_cell(CELLS[2]).len(), 8);
    assert_eq!(candidates_for_cell(CELLS[3]).len(), 8);
    for cell in [CELLS[2], CELLS[3]] {
        assert!(
            candidates_for_cell(cell)
                .iter()
                .all(|candidate| candidate.module != ModuleKind::TriadSm120)
        );
    }
    assert!(
        candidates_for_cell(CELLS[17])
            .iter()
            .all(|candidate| !candidate.is_split_k())
    );
}

#[test]
fn completion_schema_is_stable_and_machine_readable() {
    let record = format!(
        "{{\"schema\":\"{SCHEMA}\",\"cells\":{},\"discovery_windows_per_order\":{DISCOVERY_WINDOWS},\"final_windows_per_order\":{FINAL_WINDOWS},\"passed\":true}}",
        CELLS.len()
    );
    assert_eq!(
        record,
        concat!(
            "{\"schema\":\"MambaBiSm120Tf32SelectorQualificationV1\",",
            "\"cells\":18,\"discovery_windows_per_order\":21,",
            "\"final_windows_per_order\":101,\"passed\":true}"
        )
    );
}

#[test]
fn cell_artifact_schema_has_exact_candidate_and_stat_counts() {
    let order = OrderStats {
        scalar_median_us: 4.0,
        candidate_median_us: 3.0,
        median_speedup: 1.3,
        p05_speedup: 1.1,
    };
    let pair = PairStats {
        ab: order,
        ba: order,
    };
    let raw = Samples::new(
        OrderSamples {
            pairs: vec![PairedTiming {
                scalar_us: 4.0,
                candidate_us: 2.0,
            }],
        },
        OrderSamples {
            pairs: vec![PairedTiming {
                scalar_us: 4.0,
                candidate_us: 2.0,
            }],
        },
    );
    let candidates = candidates(ResolvedGemmOp::Nn)
        .iter()
        .map(|candidate| CandidateResult {
            symbol: candidate.symbol,
            numeric_family: candidate.numeric_family(),
            request_digest: "11".repeat(32),
            launch_digest: "22".repeat(32),
            route_identity_json: "{}".into(),
            eager_iterations: 128,
            graph_iterations: 256,
            eager_calibration_preflight: "eager-calibration".into(),
            eager_timed_preflight: "eager-timed".into(),
            eager_postflight: "eager-post".into(),
            graph_calibration_preflight: "graph-calibration".into(),
            graph_timed_preflight: "graph-timed".into(),
            graph_postflight: "graph-post".into(),
            discovery_eager_median: 1.1,
            discovery_graph_median: 1.2,
            final_eager_median: 1.3,
            final_eager_p05: 1.01,
            final_graph_median: 1.4,
            final_graph_p05: 1.02,
            discovery_eager_orders: pair,
            discovery_graph_orders: pair,
            final_eager_orders: pair,
            final_graph_orders: pair,
            discovery_eager_samples: raw.clone(),
            discovery_graph_samples: raw.clone(),
            final_eager_samples: raw.clone(),
            final_graph_samples: raw.clone(),
        })
        .collect();
    let json = CellResult {
        cell: CELLS[0],
        specialized_identity_json: "{}".into(),
        candidates,
        excluded_candidates: vec![CandidateExclusion {
            symbol: "excluded",
            reason: "test_reason",
        }],
        discovery_winners: ["route2", "route2"],
        final_winners: ["route2", "route2"],
        chosen_selection: "route2",
        qualified: true,
        admitted: true,
    }
    .json();

    assert!(json.starts_with(&format!("{{\"schema\":\"{SCHEMA}\"")));
    assert!(json_delimiters_are_balanced(&json));
    assert_eq!(json.matches("\"symbol\":").count(), 15);
    assert_eq!(json.matches("\"discovery\":").count(), 14);
    assert_eq!(json.matches("\"final\":").count(), 14);
    assert_eq!(json.matches("\"order_stats\":").count(), 14);
    assert_eq!(json.matches("\"raw_samples\":").count(), 14);
    assert_eq!(json.matches("\"qualification_identity\":").count(), 14);
    assert_eq!(json.matches("\"specialized_identity\":").count(), 1);
    assert_eq!(json.matches("\"iterations\":").count(), 14);
    assert_eq!(json.matches("\"ab\":").count(), 112);
    assert_eq!(json.matches("\"ba\":").count(), 112);
    assert_eq!(json.matches("\"scalar_us\":").count(), 112);
    assert_eq!(json.matches("\"candidate_us\":").count(), 112);
    assert_eq!(json.matches("\"speedup\":").count(), 112);
    assert!(json.contains("{\"scalar_us\":4,\"candidate_us\":2,\"speedup\":2}"));
    assert_eq!(json.matches("_median_speedup\"").count(), 56);
    assert_eq!(json.matches("_p05_speedup\"").count(), 28);
    assert!(json.contains("\"dims\":[64,384,1536]"));
    assert!(json.contains("\"epilogue\":{\"alpha\":1,\"beta\":0,\"bias\":true}"));
    assert!(json.contains(
        "\"excluded_candidates\":[{\"symbol\":\"excluded\",\"reason\":\"test_reason\"}]"
    ));
    assert!(json.contains("\"discovery_winners\":{\"eager\":\"route2\",\"graph\":\"route2\"}"));
    assert!(json.contains("\"final_winners\":{\"eager\":\"route2\",\"graph\":\"route2\"}"));
    assert!(json.contains("\"chosen_selection\":\"route2\""));
    assert!(json.contains("\"qualified\":true,\"admitted\":true"));
}

#[test]
fn split_numeric_gate_accepts_rounding_drift_and_rejects_corruption() {
    let reference = [1.0_f32.to_bits(), (-3.5_f32).to_bits()];
    let close = [1.001_f32.to_bits(), (-3.501_f32).to_bits()];
    assert!(check_numeric_accuracy(&close, &reference).is_ok());
    let corrupt = [f32::INFINITY.to_bits(), (-3.5_f32).to_bits()];
    assert!(check_numeric_accuracy(&corrupt, &reference).is_err());
    assert!(check_numeric_accuracy(&reference[..1], &reference).is_err());
}
