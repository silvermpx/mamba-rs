#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    D768_IN_FUSED_SYMBOL, D768_OUT_RAW_SYMBOL, PRISM_RAW_SYMBOL, PhysicalQualificationF32Epilogue,
    PhysicalQualificationRequest, PhysicalQualificationRoute, Sm89ExactF32TnRoute,
    qualify_physical_launch,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ModuleKind, ResolvedGemmOp};

const CASES: [(Sm89ExactF32TnRoute, (usize, usize, usize), [&str; 2]); 3] = [
    (
        Sm89ExactF32TnRoute::D768InDualChunkFused,
        (2_048, 768, 3_072),
        ["gemm_bi_transpose_f32_32x16_d768_v1", D768_IN_FUSED_SYMBOL],
    ),
    (
        Sm89ExactF32TnRoute::D768OutDirectBk16,
        (2_048, 1_536, 768),
        [D768_OUT_RAW_SYMBOL, "gemm_bi_splitm_reduce"],
    ),
    (
        Sm89ExactF32TnRoute::PrismDirectBk16,
        (4_621, 384, 1_928),
        [PRISM_RAW_SYMBOL, "gemm_bi_splitm_reduce"],
    ),
];

fn runtime() -> Result<GpuCtx, String> {
    let device = GpuDevice::new(0)?;
    let ctx = GpuCtx::new(&device)?;
    let route = ctx.gemm_route();
    if route.device.compute_capability != (8, 9) || route.device.multiprocessor_count != 142 {
        return Err(format!(
            "SM89 exact-F32 TN qualification requires CC8.9/142SM, got {:?}/{}SM",
            route.device.compute_capability, route.device.multiprocessor_count
        ));
    }
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_bi_tensor_cores(false);
    ctx.set_fast_gemm(false);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
    Ok(ctx)
}

fn request(
    route: PhysicalQualificationRoute,
    dims: (usize, usize, usize),
) -> PhysicalQualificationRequest {
    PhysicalQualificationRequest::contiguous_f32(
        ResolvedGemmOp::Tn,
        dims,
        route,
        PhysicalQualificationF32Epilogue::new(1.0, 1.0, false),
    )
}

fn finite_words(len: usize, salt: u32) -> Vec<u32> {
    (0..len)
        .map(|index| {
            let mixed = (index as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223 ^ salt);
            let signed = (mixed % 31) as i32 - 15;
            (signed as f32 * (1.0 / 256.0)).to_bits()
        })
        .collect()
}

fn seeded_words(dims: (usize, usize, usize), salt: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (m, k, n) = dims;
    (
        finite_words(k * n, salt ^ 0x11),
        finite_words(m * k, salt ^ 0x23),
        finite_words(m * n, salt ^ 0x47),
    )
}

fn node_symbols(
    launch: &mamba_rs::mamba_ssm::gpu::gemm_bi_triad::QualifiedPhysicalLaunch<'_>,
) -> Vec<&'static str> {
    launch
        .evidence()
        .nodes()
        .iter()
        .map(|node| node.symbol)
        .collect()
}

#[test]
#[ignore = "requires an exclusive SM89 GPU; root is the sole GPU owner"]
fn sm89_exact_f32_large_tn_forced_correctness_and_actual_auto_admission() -> Result<(), String> {
    let expect_auto = std::env::var("MAMBA_SM89_EXACT_F32_EXPECT_AUTO")
        .map(|value| value == "1")
        .unwrap_or(false);
    let forced_ctx = runtime()?;
    let auto_ctx = runtime()?;

    for (index, (route, dims, expected_symbols)) in CASES.into_iter().enumerate() {
        let forced_request = request(
            PhysicalQualificationRoute::Sm89ExactF32TnForced(route),
            dims,
        );
        let mut forced = qualify_physical_launch(&forced_ctx, forced_request)?;
        let forced_symbols = node_symbols(&forced);
        if forced_symbols != expected_symbols
            || !forced.evidence().eager_graph_equal()
            || forced.evidence().nodes().iter().any(|node| {
                node.symbol == route.symbol() && node.module_kind != ModuleKind::TriadSm89ExactF32
            })
        {
            return Err(format!(
                "forced {route:?} physical identity changed: {:?}",
                forced.evidence().nodes()
            ));
        }

        let seed = seeded_words(dims, 0x5a17 ^ index as u32);
        forced.upload_exact_unbiased_f32_words(&forced_ctx, &seed.0, &seed.1, &seed.2)?;
        let original_inputs = forced.f32_operand_bits(&forced_ctx)?;
        forced.measure_prevalidated_forced_eager_window_ms(&forced_ctx, 1)?;
        let eager_bits = forced.f32_output_bits(&forced_ctx)?;
        if forced.f32_operand_bits(&forced_ctx)? != original_inputs {
            return Err(format!("forced {route:?} eager launch modified an input"));
        }
        forced.validate_red_zones(&forced_ctx)?;

        forced.upload_exact_unbiased_f32_words(&forced_ctx, &seed.0, &seed.1, &seed.2)?;
        forced.measure_graph_window_ms(&forced_ctx, 1)?;
        let graph_bits = forced.f32_output_bits(&forced_ctx)?;
        if graph_bits != eager_bits {
            return Err(format!("forced {route:?} eager/graph bits differ"));
        }
        forced.validate_red_zones(&forced_ctx)?;
        drop(forced);

        let auto_request = request(
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
            dims,
        );
        let mut auto = qualify_physical_launch(&auto_ctx, auto_request)?;
        let auto_symbols = node_symbols(&auto);
        if expect_auto && auto_symbols != expected_symbols {
            return Err(format!(
                "actual AUTO has not admitted {route:?}: expected {expected_symbols:?}, got {auto_symbols:?}"
            ));
        }
        auto.upload_exact_unbiased_f32_words(&auto_ctx, &seed.0, &seed.1, &seed.2)?;
        auto.measure_eager_window_ms(&auto_ctx, 1)?;
        let auto_bits = auto.f32_output_bits(&auto_ctx)?;
        if auto_bits != eager_bits {
            return Err(format!("forced {route:?} differs bitwise from actual AUTO"));
        }
        auto.validate_red_zones(&auto_ctx)?;
        println!(
            "{{\"schema\":\"MambaTriadSm89ExactF32TnQualificationV1\",\"route\":\"{:?}\",\"dims\":[{},{},{}],\"forced_symbols\":{:?},\"actual_auto_symbols\":{:?},\"expect_auto\":{},\"exact\":true,\"eager_graph_equal\":true,\"guards\":true}}",
            route, dims.0, dims.1, dims.2, forced_symbols, auto_symbols, expect_auto
        );
        drop(auto);
    }
    Ok(())
}
