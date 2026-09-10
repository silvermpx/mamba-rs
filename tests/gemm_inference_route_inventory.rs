#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GemmMode, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_inference::inference_forward;
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

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_triad_public_half_matvec_records_both_tensor_core_states() {
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{PhysicalGemmBackend, PolicyDtype};
    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&device).unwrap();
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    for tc in [false, true] {
        ctx.set_bi_tensor_cores(tc);
        for input_dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for output_dtype in [input_dtype, WeightDtype::F32] {
                let x = DtypedBuf::zeros(&ctx.stream, 3 * 37, input_dtype).unwrap();
                let w = DtypedBuf::zeros(&ctx.stream, 37 * 16, input_dtype).unwrap();
                let y = DtypedBuf::zeros(&ctx.stream, 3 * 16, output_dtype).unwrap();
                x.upload_f32(&ctx.stream, &vec![1.0; 3 * 37]).unwrap();
                w.upload_f32(&ctx.stream, &vec![1.0; 37 * 16]).unwrap();
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
                gpu_gemm_typed_forward_raw(&ctx, output, input, weight, None, (3, 37, 16)).unwrap();
                let mut expected = vec![0.0; 3 * 16];
                y.download_f32(&ctx.stream, &mut expected).unwrap();
                let trace = ctx
                    .record_eager_gemm_trace(|| {
                        gpu_gemm_typed_forward_raw(&ctx, output, input, weight, None, (3, 37, 16))
                    })
                    .unwrap();
                assert_eq!(trace.routes().len(), 1);
                let route = trace.routes()[0];
                assert_eq!(route.backend, PhysicalGemmBackend::FixedMatvecEightWarpV1);
                assert_eq!(
                    route.dtype,
                    if input_dtype == WeightDtype::Bf16 {
                        PolicyDtype::Bf16
                    } else {
                        PolicyDtype::F16
                    }
                );
                assert_eq!(route.shape, (3, 37, 16));
                assert_eq!(route.strides, (37, 16, 16));
                assert_eq!(route.tile, (1, 32));
                assert_eq!(route.bk, 0);
                assert_eq!(route.launch.grid_dim, (1, 3, 1));
                assert_eq!(route.launch.shared_mem_bytes, 80);
                let mut actual = vec![0.0; 3 * 16];
                y.download_f32(&ctx.stream, &mut actual).unwrap();
                assert_eq!(actual, expected);
                assert!(actual.iter().all(|value| *value == 37.0));
            }
        }
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn deterministic_inference_direct_empty_output_has_no_terminal_record() {
    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&device).unwrap();
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    // Canonical F32 forwarding rejects empty output axes before dispatch.
    // The direct Inference adapter instead preserves terminal no-op behavior.
    let f32_null = TypedPtr {
        ptr: 0,
        dtype: WeightDtype::F32,
    };
    for shape in [(0, 37, 96), (3, 37, 0)] {
        let trace = ctx
            .record_eager_gemm_trace(|| {
                let error =
                    gpu_gemm_typed_forward_raw(&ctx, f32_null, f32_null, f32_null, None, shape)
                        .expect_err("canonical F32 forwarding must reject empty output axes");
                assert!(error.contains("output axes must be positive"), "{error}");
                Ok(())
            })
            .unwrap();
        assert!(trace.routes().is_empty());
    }
    for (input_dtype, output_dtype) in [
        (WeightDtype::F32, WeightDtype::F32),
        (WeightDtype::Bf16, WeightDtype::Bf16),
        (WeightDtype::F16, WeightDtype::F16),
        (WeightDtype::Bf16, WeightDtype::F32),
        (WeightDtype::F16, WeightDtype::F32),
    ] {
        let input = TypedPtr {
            ptr: 0,
            dtype: input_dtype,
        };
        let output = TypedPtr {
            ptr: 0,
            dtype: output_dtype,
        };
        for shape in [(0, 37, 96), (3, 37, 0)] {
            let trace = ctx
                .record_eager_gemm_trace(|| {
                    inference_forward(&ctx, output, input, input, None, shape).map(|_| ())
                })
                .unwrap();
            assert!(trace.routes().is_empty());
        }
    }
}
