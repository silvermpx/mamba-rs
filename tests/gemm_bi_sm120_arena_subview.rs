//! A matrix sliced out of a flat arena starts inside a row of the
//! allocation's grid. The SM120 tensor-map route has to serve such an
//! operand from its own first element, with the same kernel and the same
//! bits as the standalone operand, instead of refusing the launch.

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GemmMode, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

const M: usize = 4621;
const K: usize = 1928;
const N: usize = 384;
/// Elements of padding in front of the sliced operand: 128 bytes of bf16,
/// which keeps the tensor-map alignment and lands inside the first row of
/// both operands, so their columns run past the allocation's row grid.
const PAD: usize = 64;

fn synth(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn upload(ctx: &GpuCtx, host: &[f32]) -> DtypedBuf {
    let buffer =
        DtypedBuf::zeros(&ctx.stream, host.len(), WeightDtype::Bf16).expect("allocate bf16");
    buffer
        .upload_f32(&ctx.stream, host)
        .expect("upload bf16 operand");
    buffer
}

fn padded(host: &[f32]) -> Vec<f32> {
    let mut arena = vec![0.0f32; PAD + host.len()];
    arena[PAD..].copy_from_slice(host);
    arena
}

#[test]
fn sm120_half_route_serves_operands_sliced_from_an_arena() {
    let device = GpuDevice::new(0).expect("CUDA device");
    if device.compute_capability != (12, 0) {
        println!("skipped: the arena subview check needs an SM120 board");
        return;
    }
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("deterministic GEMM mode");
    ctx.route_controls().set_family(BiGemmFamily::Triad);
    ctx.route_controls().set_tensor_cores(true);

    let x_host = synth(M * K, 0x51ce_a11e);
    let w_host = synth(K * N, 0xa2ea_0b01);
    let x = upload(&ctx, &x_host);
    let w = upload(&ctx, &w_host);
    let x_arena = upload(&ctx, &padded(&x_host));
    let w_arena = upload(&ctx, &padded(&w_host));
    let y = DtypedBuf::zeros(&ctx.stream, M * N, WeightDtype::Bf16).expect("allocate output");
    let slice_offset_bytes = (PAD * 2) as u64;

    let run = |x_ptr: u64, w_ptr: u64| -> (Vec<u32>, &'static str) {
        let typed = |ptr| TypedPtr {
            ptr,
            dtype: WeightDtype::Bf16,
        };
        let trace = ctx
            .record_eager_gemm_trace(|| {
                gpu_gemm_typed_forward_raw(
                    &ctx,
                    typed(y.cached_ptr()),
                    typed(x_ptr),
                    typed(w_ptr),
                    None,
                    (M, K, N),
                )
            })
            .expect("typed forward on the SM120 route");
        let routes = trace.routes();
        assert_eq!(routes.len(), 1, "one launch expected: {routes:?}");
        let mut output = vec![0.0f32; M * N];
        y.download_f32(&ctx.stream, &mut output)
            .expect("download output");
        (
            output.iter().map(|v| v.to_bits()).collect(),
            routes[0].symbol,
        )
    };

    let (reference, symbol) = run(x.cached_ptr(), w.cached_ptr());
    assert!(
        symbol.contains("sm120_tma"),
        "the standalone operands did not take the SM120 tensor-map route: {symbol}"
    );
    let cases = [
        (
            "A sliced",
            x_arena.cached_ptr() + slice_offset_bytes,
            w.cached_ptr(),
        ),
        (
            "B sliced",
            x.cached_ptr(),
            w_arena.cached_ptr() + slice_offset_bytes,
        ),
        (
            "both sliced",
            x_arena.cached_ptr() + slice_offset_bytes,
            w_arena.cached_ptr() + slice_offset_bytes,
        ),
    ];
    for (label, x_ptr, w_ptr) in cases {
        let (bits, sliced_symbol) = run(x_ptr, w_ptr);
        assert_eq!(sliced_symbol, symbol, "{label}: the route changed");
        assert!(bits == reference, "{label}: the output bits differ");
    }
}
