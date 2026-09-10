#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GemmMode, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;

fn assert_public_inference_nn_is_inventoried(input_dtype: WeightDtype, output_dtype: WeightDtype) {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_gemm_mode(GemmMode::Deterministic)
        .expect("select deterministic GEMM mode");
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);

    let x = DtypedBuf::zeros(&ctx.stream, 3 * 37, input_dtype).expect("allocate X[3,37]");
    let w = DtypedBuf::zeros(&ctx.stream, 37 * 96, input_dtype).expect("allocate W[37,96]");
    let y = DtypedBuf::zeros(&ctx.stream, 3 * 96, output_dtype).expect("allocate Y[3,96]");
    x.upload_f32(&ctx.stream, &vec![1.0; 3 * 37])
        .expect("upload X");
    w.upload_f32(&ctx.stream, &vec![1.0; 37 * 96])
        .expect("upload W");
    let input = TypedPtr {
        ptr: x.cached_ptr(),
        dtype: input_dtype,
    };
    let weight = TypedPtr {
        ptr: w.cached_ptr(),
        dtype: input_dtype,
    };
    let output = TypedPtr {
        ptr: y.cached_ptr(),
        dtype: output_dtype,
    };

    let trace = ctx
        .record_eager_gemm_trace(|| {
            gpu_gemm_typed_forward_raw(&ctx, output, input, weight, None, (3, 37, 96))
        })
        .expect("public deterministic Inference forward");
    assert!(
        !trace.routes().is_empty(),
        "Inference NN launch must be inventoried"
    );
    assert!(
        trace.routes().iter().all(|route| {
            route.op == ResolvedGemmOp::Nn
                && route.shape == (3, 37, 96)
                && route.strides == (37, 96, 96)
        }),
        "unexpected Inference routes: {:?}",
        trace.routes()
    );

    let mut actual = vec![0.0; 3 * 96];
    y.download_f32(&ctx.stream, &mut actual)
        .expect("download Y");
    assert!(
        actual
            .iter()
            .all(|value| value.to_bits() == 37.0f32.to_bits())
    );
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_inference_public_f32_forward_records_nn_route() {
    assert_public_inference_nn_is_inventoried(WeightDtype::F32, WeightDtype::F32);
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_inference_public_bf16_forward_records_nn_route() {
    assert_public_inference_nn_is_inventoried(WeightDtype::Bf16, WeightDtype::Bf16);
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_inference_public_bf16_to_f32_forward_records_nn_route() {
    assert_public_inference_nn_is_inventoried(WeightDtype::Bf16, WeightDtype::F32);
}
