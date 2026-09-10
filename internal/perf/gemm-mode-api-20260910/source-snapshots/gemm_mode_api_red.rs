#![cfg(feature = "cuda")]

use cudarc::cublas::sys::{cublasGetMathMode, cublasMath_t, cublasStatus_t};
use mamba_rs::mamba_ssm::gpu::context::{
    BiGemmFamily, F32TriadPolicy, GemmMode, GpuCtx, HalfTriadPolicy,
};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;

#[test]
fn gemm_mode_names_and_default() {
    use mamba_rs::mamba_ssm::gpu::context::GemmMode;
    assert_eq!(GemmMode::default(), GemmMode::Deterministic);
    for (name, mode) in [
        ("deterministic", GemmMode::Deterministic),
        ("cublas-fast", GemmMode::CublasFast),
        ("cublas-pedantic", GemmMode::CublasPedantic),
    ] {
        assert_eq!(GemmMode::parse_env_value(name).unwrap(), mode);
        assert_eq!(mode.as_str(), name);
    }
    assert_eq!(
        GemmMode::parse_env_value(" \tdeterministic\n").unwrap(),
        GemmMode::Deterministic
    );
    for bad in ["", "auto", "fast", "Deterministic", "cublas"] {
        assert!(GemmMode::parse_env_value(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn gpu_module_reexports_the_canonical_gemm_mode() {
    let mode: mamba_rs::mamba_ssm::gpu::GemmMode = GemmMode::CublasPedantic;
    assert_eq!(mode, GemmMode::CublasPedantic);
}

fn actual_cublas_math(ctx: &GpuCtx) -> cublasMath_t {
    let mut math = cublasMath_t::CUBLAS_DEFAULT_MATH;
    let status = unsafe { cublasGetMathMode(*ctx.blas.handle(), &mut math) };
    assert_eq!(
        status,
        cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "query cuBLAS handle math"
    );
    math
}

fn assert_mode(ctx: &GpuCtx, expected: GemmMode) {
    let (batch_invariant, fast_gemm, tf32, handle_math) = match expected {
        GemmMode::Deterministic => (true, false, false, cublasMath_t::CUBLAS_PEDANTIC_MATH),
        GemmMode::CublasFast => (false, true, true, cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH),
        GemmMode::CublasPedantic => (false, false, false, cublasMath_t::CUBLAS_PEDANTIC_MATH),
    };
    let policy = ctx.gemm_policy();
    assert_eq!(ctx.gemm_mode(), expected, "canonical mode");
    assert_eq!(
        ctx.batch_invariant(),
        batch_invariant,
        "batch-invariant mapping"
    );
    assert_eq!(ctx.fast_gemm(), fast_gemm, "fast-GEMM mapping");
    assert_eq!(ctx.tf32(), tf32, "TF32 mapping");
    assert_eq!(
        policy.batch_invariant, batch_invariant,
        "policy batch-invariant mapping"
    );
    assert_eq!(policy.fast_gemm, fast_gemm, "policy fast-GEMM mapping");
    assert_eq!(policy.cublas_tf32, tf32, "policy TF32 mapping");
    assert_eq!(
        actual_cublas_math(ctx),
        handle_math,
        "cuBLAS handle math mapping"
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn gpu_context_constructors_use_explicit_modes_and_deterministic_defaults() {
    let device = GpuDevice::new(0).expect("CUDA device");

    let ctx = GpuCtx::new(&device).expect("default GPU context");
    assert_mode(&ctx, GemmMode::Deterministic);
    assert_eq!(ctx.bi_gemm_family(), BiGemmFamily::Triad);
    assert!(
        ctx.bi_tensor_cores(),
        "deterministic tensor-core permission default"
    );
    assert_eq!(ctx.f32_triad_policy(), F32TriadPolicy::ExactScalarFmaV1);
    assert_eq!(ctx.half_triad_policy(), HalfTriadPolicy::TiledParityV1);
    drop(ctx);

    let ctx =
        GpuCtx::new_with_mode(&device, GemmMode::CublasFast).expect("explicit Fast GPU context");
    assert_mode(&ctx, GemmMode::CublasFast);
    drop(ctx);

    let ctx = GpuCtx::new_with_state_cap_and_mode(&device, 96, GemmMode::CublasPedantic)
        .expect("explicit Pedantic GPU context with state capacity");
    assert_mode(&ctx, GemmMode::CublasPedantic);
    assert_eq!(ctx.state_cap(), 96);
}

#[test]
#[ignore = "needs a CUDA device"]
fn gpu_context_supports_all_nine_canonical_mode_transitions() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let modes = [
        GemmMode::Deterministic,
        GemmMode::CublasFast,
        GemmMode::CublasPedantic,
    ];

    for from in modes {
        ctx.set_gemm_mode(from).expect("select source mode");
        assert_mode(&ctx, from);
        for to in modes {
            ctx.set_gemm_mode(from).expect("restore source mode");
            ctx.set_gemm_mode(to).expect("transition to target mode");
            assert_mode(&ctx, to);
        }
    }
}

#[test]
#[ignore = "needs a CUDA device"]
#[expect(
    deprecated,
    reason = "this regression freezes the legacy migration adapters"
)]
fn legacy_mode_adapters_preserve_the_documented_call_order() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");

    ctx.set_batch_invariant(true);
    ctx.set_fast_gemm(false);
    assert_mode(&ctx, GemmMode::Deterministic);

    ctx.set_gemm_mode(GemmMode::CublasPedantic)
        .expect("reset to Pedantic");
    ctx.set_fast_gemm(false);
    ctx.set_batch_invariant(true);
    assert_mode(&ctx, GemmMode::Deterministic);

    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("reset to Deterministic");
    ctx.set_fast_gemm(true);
    ctx.set_batch_invariant(false);
    assert_mode(&ctx, GemmMode::CublasFast);

    ctx.set_gemm_mode(GemmMode::CublasPedantic)
        .expect("reset to Pedantic");
    ctx.set_batch_invariant(false);
    ctx.set_fast_gemm(true);
    assert_mode(&ctx, GemmMode::CublasFast);

    ctx.disable_tf32();
    assert_mode(&ctx, GemmMode::CublasPedantic);
    ctx.set_batch_invariant(true);
    ctx.disable_tf32();
    assert_mode(&ctx, GemmMode::Deterministic);
}

#[test]
#[ignore = "needs a CUDA device"]
fn vendor_mode_round_trip_preserves_custom_deterministic_policy() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_bi_tensor_cores(true);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    ctx.set_half_triad_policy(HalfTriadPolicy::AllowStreamKFixedOrderV1);

    for vendor_mode in [GemmMode::CublasFast, GemmMode::CublasPedantic] {
        ctx.set_gemm_mode(vendor_mode).expect("select vendor mode");
        assert_mode(&ctx, vendor_mode);
        ctx.set_gemm_mode(GemmMode::Deterministic)
            .expect("restore Deterministic mode");
        assert_mode(&ctx, GemmMode::Deterministic);
        assert_eq!(ctx.bi_gemm_family(), BiGemmFamily::Inference);
        assert!(ctx.bi_tensor_cores());
        assert_eq!(
            ctx.f32_triad_policy(),
            F32TriadPolicy::AllowDeterministicTf32V1
        );
        assert_eq!(
            ctx.half_triad_policy(),
            HalfTriadPolicy::AllowStreamKFixedOrderV1
        );
    }
}
