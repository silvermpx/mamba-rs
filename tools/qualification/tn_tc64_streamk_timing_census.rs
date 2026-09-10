//! Timing survey of the tc64 TN dW body over its two schedules (the tiled
//! kernel and the stream-K twin) as graph replays, one JSON line per cell.
//! The automatic SM89 rule was set from these numbers. Manual instrument:
//! build with `--features cuda,qualification` and run the ignored test on
//! the measured board.

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad;

struct Ctx {
    ctx: GpuCtx,
}
impl Ctx {
    fn new() -> Self {
        let device = GpuDevice::new(0).expect("gpu");
        let ctx = GpuCtx::new(&device).expect("ctx");
        Self { ctx }
    }
}
/// The portable module composes the tc64 TN stream-K fragment on every
/// sm80-family board except CC 12.x, whose boards run the SM120 stream-K
/// kernel; the forced stream-K tests have nothing to launch there.
fn portable_module_composes_streamk() -> bool {
    let device = GpuDevice::new(0).expect("device 0");
    if device.compute_capability.0 == 12 {
        eprintln!(
            "skip: the portable stream-K fragment is not composed for CC {:?}",
            device.compute_capability
        );
        return false;
    }
    true
}
/// Timing survey of the tc64 TN dW body over its two schedules on this
/// device: the tiled kernel and the stream-K twin, forced through the
/// physical qualification facade and timed as graph replays. Prints one
/// JSON line per cell; the automatic SM89 rule is set from these numbers.
#[test]
#[ignore = "timing survey; run by hand on the measured board"]
fn tn_tc64_streamk_versus_tiled_timing_census() {
    use gemm_bi_triad::{
        PhysicalQualificationRequest, PhysicalQualificationRoute, TcTile, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;

    if !portable_module_composes_streamk() {
        return;
    }

    const SHAPES: [(&str, (usize, usize, usize)); 22] = [
        ("sq64", (64, 64, 64)),
        ("underfill", (256, 512, 384)),
        ("large", (2048, 3072, 768)),
        ("large_deep", (4096, 3072, 1536)),
        ("rect_wide", (512, 3072, 768)),
        ("rect_tall", (4096, 512, 768)),
        ("thin_rows", (16, 512, 2048)),
        ("thin_cols", (512, 16, 2048)),
        ("thin_rows_tail", (49, 65, 129)),
        ("thin_cols_tail", (65, 49, 129)),
        ("all_tail", (129, 131, 100)),
        ("split_candidate", (128, 8192, 128)),
        ("d128_in_proj", (1024, 128, 512)),
        ("d128_out_proj", (1024, 256, 128)),
        ("d768_in_proj", (2048, 768, 3072)),
        ("d768_out_proj", (2048, 1536, 768)),
        ("prism_in_proj", (4621, 384, 1928)),
        ("prism_out_proj", (4621, 768, 384)),
        ("prism_input_proj", (4621, 1024, 384)),
        ("batch_in_proj", (10400, 384, 1536)),
        ("batch_input_proj", (10400, 384, 384)),
        ("batch_out_proj", (10400, 768, 384)),
    ];
    const WINDOWS: usize = 9;
    const ITERATIONS: usize = 20;

    let t = Ctx::new();
    let multiprocessors = mamba_rs::mamba_ssm::gpu::device::GpuDevice::new(0)
        .expect("query device 0")
        .multiprocessor_count();
    // Every cell captures a graph, and the context's scratch cannot grow
    // after a capture: size the whole suite before the first cell.
    let suite = SHAPES
        .iter()
        .flat_map(|(_, dims)| {
            [WeightDtype::Bf16, WeightDtype::F16]
                .into_iter()
                .flat_map(move |dtype| {
                    [TcTile::Tile64, TcTile::Tile128, TcTile::Tile64StreamK]
                        .into_iter()
                        .map(move |tile| {
                            PhysicalQualificationRequest::contiguous(
                                ResolvedGemmOp::Tn,
                                *dims,
                                PhysicalQualificationRoute::HalfForced { dtype, tile },
                            )
                        })
                })
        })
        .collect::<Vec<_>>();
    gemm_bi_triad::presize_physical_qualification_suite(&t.ctx, &suite)
        .expect("pre-size the stream-K timing suite");
    for (name, dims) in SHAPES {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let mut row = format!(
                "{{\"schema\":\"MambaBiTnStreamKCensusV1\",\"shape\":\"{name}\",\"m\":{},\"k\":{},\"n\":{},\"dtype\":\"{}\",\"multiprocessors\":{multiprocessors}",
                dims.0,
                dims.1,
                dims.2,
                match dtype {
                    WeightDtype::Bf16 => "bf16",
                    WeightDtype::F16 => "f16",
                    WeightDtype::F32 => "f32",
                }
            );
            for (label, tile) in [
                ("tile64", TcTile::Tile64),
                ("tile128", TcTile::Tile128),
                ("tile64_streamk", TcTile::Tile64StreamK),
            ] {
                let request = PhysicalQualificationRequest::contiguous(
                    ResolvedGemmOp::Tn,
                    dims,
                    PhysicalQualificationRoute::HalfForced { dtype, tile },
                );
                let physical = match qualify_physical_launch(&t.ctx, request) {
                    Ok(physical) => physical,
                    Err(error) => {
                        row.push_str(&format!(
                            ",\"{label}_error\":\"{}\"",
                            error.replace('"', "'")
                        ));
                        continue;
                    }
                };
                let grid = physical.evidence().nodes()[0].launch.grid_dim.0;
                let mut best = f64::INFINITY;
                for _ in 0..WINDOWS {
                    let window = physical
                        .measure_graph_window_ms(&t.ctx, ITERATIONS)
                        .expect("measure graph window");
                    best = best.min(window * 1000.0 / ITERATIONS as f64);
                }
                row.push_str(&format!(
                    ",\"{label}_grid\":{grid},\"{label}_us\":{best:.3}"
                ));
            }
            row.push('}');
            println!("{row}");
        }
    }
}
