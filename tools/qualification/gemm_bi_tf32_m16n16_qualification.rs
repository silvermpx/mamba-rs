#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    PhysicalQualificationRequest, PhysicalQualificationRoute, QualifiedPhysicalLaunch,
    Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile,
    presize_physical_qualification_suite, qualify_physical_launch,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;

const WINDOWS: usize = 101;
const TARGET_WINDOW_MS: f64 = 5.0;
const WARMUPS: usize = 128;

#[derive(Clone, Copy)]
struct Cell {
    id: &'static str,
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
    invocation_weight: f64,
}

const CELLS: [Cell; 12] = [
    Cell {
        id: "nn_m64_k1536_n384",
        op: ResolvedGemmOp::Nn,
        dims: (64, 1536, 384),
        invocation_weight: 21.0,
    },
    Cell {
        id: "nn_m64_k384_n1536",
        op: ResolvedGemmOp::Nn,
        dims: (64, 384, 1536),
        invocation_weight: 21.0,
    },
    Cell {
        id: "nn_m64_k833_n384",
        op: ResolvedGemmOp::Nn,
        dims: (64, 833, 384),
        invocation_weight: 10.5,
    },
    Cell {
        id: "nn_m64_k384_n101",
        op: ResolvedGemmOp::Nn,
        dims: (64, 384, 101),
        invocation_weight: 10.5,
    },
    Cell {
        id: "nt_m64_k384_n1536",
        op: ResolvedGemmOp::Nt,
        dims: (64, 384, 1536),
        invocation_weight: 9.0,
    },
    Cell {
        id: "nt_m64_k1536_n384",
        op: ResolvedGemmOp::Nt,
        dims: (64, 1536, 384),
        invocation_weight: 9.0,
    },
    Cell {
        id: "tn_m64_k384_n1536",
        op: ResolvedGemmOp::Tn,
        dims: (64, 384, 1536),
        invocation_weight: 6.0,
    },
    Cell {
        id: "tn_m64_k1536_n384",
        op: ResolvedGemmOp::Tn,
        dims: (64, 1536, 384),
        invocation_weight: 6.0,
    },
    Cell {
        id: "nn_m64_k256_n256",
        op: ResolvedGemmOp::Nn,
        dims: (64, 256, 256),
        invocation_weight: 8.0,
    },
    Cell {
        id: "nn_m64_k192_n256",
        op: ResolvedGemmOp::Nn,
        dims: (64, 192, 256),
        invocation_weight: 3.0,
    },
    Cell {
        id: "nn_m64_k256_n1024",
        op: ResolvedGemmOp::Nn,
        dims: (64, 256, 1024),
        invocation_weight: 1.5,
    },
    Cell {
        id: "nn_m64_k1024_n256",
        op: ResolvedGemmOp::Nn,
        dims: (64, 1024, 256),
        invocation_weight: 1.5,
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

fn route(tile: Tf32PortableTile) -> Tf32PhysicalRoute {
    Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
        tile,
        stages: Tf32PortableStages::S4,
    })
}

fn request(cell: Cell, tile: Tf32PortableTile) -> PhysicalQualificationRequest {
    PhysicalQualificationRequest::contiguous(
        cell.op,
        cell.dims,
        PhysicalQualificationRoute::Tf32Forced(route(tile)),
    )
}

fn context(device: &GpuDevice) -> Result<GpuCtx, String> {
    let ctx = GpuCtx::new(device)?;
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.route_controls().set_family(BiGemmFamily::Triad);
    ctx.route_controls()
        .set_f32_policy(F32TriadPolicy::AllowDeterministicTf32);
    Ok(ctx)
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
    Ok(total * 1000.0 / iterations as f64)
}

fn calibrate(
    launch: &mut QualifiedPhysicalLaunch<'_>,
    ctx: &GpuCtx,
    path: Path,
) -> Result<usize, String> {
    let pilot = measure(launch, ctx, path, 16)?;
    if !pilot.is_finite() || pilot <= 0.0 {
        return Err(format!("{} pilot was not positive and finite", path.name()));
    }
    Ok(((TARGET_WINDOW_MS * 1000.0 / pilot).ceil() as usize).clamp(1, 1_000_000))
}

fn percentile(values: &[f64], numerator: usize) -> Result<f64, String> {
    if values.len() != WINDOWS || numerator > 100 {
        return Err("percentile requires 101 samples and a percentile in 0..=100".into());
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.total_cmp(right));
    Ok(ordered[numerator])
}

fn assert_route(
    launch: &QualifiedPhysicalLaunch<'_>,
    symbol: &str,
    tile: (u32, u32),
) -> Result<(), String> {
    let evidence = launch.evidence();
    if !evidence.eager_graph_equal()
        || evidence.launch_count() != 1
        || evidence.single_launch_symbol() != Some(symbol)
        || evidence.single_launch_tile() != Some(tile)
    {
        return Err(format!(
            "forced physical route mismatch: {:?}",
            evidence.nodes()
        ));
    }
    Ok(())
}

fn bit_gate(
    cell: Cell,
    candidate_ctx: &GpuCtx,
    candidate: &mut QualifiedPhysicalLaunch<'_>,
    incumbent_ctx: &GpuCtx,
    incumbent: &mut QualifiedPhysicalLaunch<'_>,
) -> Result<(), String> {
    let salt = 0x006d_3136_6e38_u64 ^ cell.dims.0 as u64 ^ ((cell.dims.1 as u64) << 17);
    candidate.seed_f32_operands(candidate_ctx, salt)?;
    measure(candidate, candidate_ctx, Path::Eager, 1)?;
    let candidate_eager = candidate.f32_output_bits(candidate_ctx)?;
    candidate.seed_f32_operands(candidate_ctx, salt)?;
    measure(candidate, candidate_ctx, Path::Graph, 1)?;
    let candidate_graph = candidate.f32_output_bits(candidate_ctx)?;
    incumbent.seed_f32_operands(incumbent_ctx, salt)?;
    measure(incumbent, incumbent_ctx, Path::Eager, 1)?;
    let incumbent_eager = incumbent.f32_output_bits(incumbent_ctx)?;
    incumbent.seed_f32_operands(incumbent_ctx, salt)?;
    measure(incumbent, incumbent_ctx, Path::Graph, 1)?;
    let incumbent_graph = incumbent.f32_output_bits(incumbent_ctx)?;
    if candidate_eager != candidate_graph
        || incumbent_eager != incumbent_graph
        || candidate_eager != incumbent_eager
    {
        return Err(format!("{} M16N16/M16N32 eager/graph bits differ", cell.id));
    }
    Ok(())
}

fn run_cell(device: &GpuDevice, cell: Cell) -> Result<(), String> {
    let candidate_ctx = context(device)?;
    let incumbent_ctx = context(device)?;
    let candidate_request = request(cell, Tf32PortableTile::M16N16);
    let incumbent_request = request(cell, Tf32PortableTile::M16N32);
    presize_physical_qualification_suite(&candidate_ctx, &[candidate_request])?;
    presize_physical_qualification_suite(&incumbent_ctx, &[incumbent_request])?;
    let mut candidate = qualify_physical_launch(&candidate_ctx, candidate_request)?;
    let mut incumbent = qualify_physical_launch(&incumbent_ctx, incumbent_request)?;
    assert_route(
        &candidate,
        match cell.op {
            ResolvedGemmOp::Nn => "nn_sm80_mma_tf32_m16n16_bk32_s4",
            ResolvedGemmOp::Tn => "tn_sm80_mma_tf32_m16n16_bk32_s4",
            ResolvedGemmOp::Nt => "nt_sm80_mma_tf32_m16n16_bk32_s4",
        },
        (16, 16),
    )?;
    assert_route(
        &incumbent,
        match cell.op {
            ResolvedGemmOp::Nn => "nn_sm80_mma_tf32_m16n32_bk32_s4",
            ResolvedGemmOp::Tn => "tn_sm80_mma_tf32_m16n32_bk32_s4",
            ResolvedGemmOp::Nt => "nt_sm80_mma_tf32_m16n32_bk32_s4",
        },
        (16, 32),
    )?;
    bit_gate(
        cell,
        &candidate_ctx,
        &mut candidate,
        &incumbent_ctx,
        &mut incumbent,
    )?;

    for _ in 0..WARMUPS {
        measure(&mut candidate, &candidate_ctx, Path::Graph, 1)?;
        measure(&mut incumbent, &incumbent_ctx, Path::Graph, 1)?;
    }
    for path in [Path::Eager, Path::Graph] {
        let candidate_iterations = calibrate(&mut candidate, &candidate_ctx, path)?;
        let incumbent_iterations = calibrate(&mut incumbent, &incumbent_ctx, path)?;
        for candidate_first in [true, false] {
            let mut candidate_us = Vec::with_capacity(WINDOWS);
            let mut incumbent_us = Vec::with_capacity(WINDOWS);
            let mut ratios = Vec::with_capacity(WINDOWS);
            for _ in 0..WINDOWS {
                let (candidate_sample, incumbent_sample) = if candidate_first {
                    (
                        measure(&mut candidate, &candidate_ctx, path, candidate_iterations)?,
                        measure(&mut incumbent, &incumbent_ctx, path, incumbent_iterations)?,
                    )
                } else {
                    let incumbent_sample =
                        measure(&mut incumbent, &incumbent_ctx, path, incumbent_iterations)?;
                    let candidate_sample =
                        measure(&mut candidate, &candidate_ctx, path, candidate_iterations)?;
                    (candidate_sample, incumbent_sample)
                };
                candidate_us.push(candidate_sample);
                incumbent_us.push(incumbent_sample);
                ratios.push(candidate_sample / incumbent_sample);
            }
            println!(
                concat!(
                    "M16N16\t{}\t{:?}\t{}\t{}\t{}\t",
                    "candidate_p50_us={:.9}\tcandidate_p95_us={:.9}\t",
                    "incumbent_p50_us={:.9}\tincumbent_p95_us={:.9}\t",
                    "ratio_p50={:.9}\tratio_p95={:.9}"
                ),
                cell.id,
                cell.op,
                path.name(),
                if candidate_first {
                    "candidate_first"
                } else {
                    "incumbent_first"
                },
                cell.invocation_weight,
                percentile(&candidate_us, 50)?,
                percentile(&candidate_us, 95)?,
                percentile(&incumbent_us, 50)?,
                percentile(&incumbent_us, 95)?,
                percentile(&ratios, 50)?,
                percentile(&ratios, 95)?,
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an idle SM80+ GPU and runs paired 101-window qualification"]
fn portable_tf32_m16n16_production_qualification() -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("M16N16 performance qualification requires --release".into());
    }
    let device = GpuDevice::new(0)?;
    if device.compute_capability < (8, 0) {
        return Err("M16N16 qualification requires SM80 or newer".into());
    }
    for cell in CELLS {
        run_cell(&device, cell)?;
    }
    Ok(())
}
