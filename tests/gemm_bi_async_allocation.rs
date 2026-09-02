#![cfg(feature = "cuda")]

use cudarc::driver::sys;
use mamba_rs::mamba_ssm::gpu::blas::gpu_gemm_bi_forward_raw;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;

#[derive(Debug)]
struct PointerAttributes {
    context: sys::CUcontext,
    device_ordinal: i32,
    buffer_id: u64,
    range_start: sys::CUdeviceptr,
    range_bytes: usize,
}

fn pointer_attributes(pointer: sys::CUdeviceptr) -> PointerAttributes {
    let mut attributes = PointerAttributes {
        context: std::ptr::null_mut(),
        device_ordinal: -1,
        buffer_id: 0,
        range_start: 0,
        range_bytes: 0,
    };
    let mut names = [
        sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
        sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_BUFFER_ID,
        sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_RANGE_SIZE,
    ];
    let mut outputs = [
        std::ptr::from_mut(&mut attributes.context).cast(),
        std::ptr::from_mut(&mut attributes.device_ordinal).cast(),
        std::ptr::from_mut(&mut attributes.buffer_id).cast(),
        std::ptr::from_mut(&mut attributes.range_start).cast(),
        std::ptr::from_mut(&mut attributes.range_bytes).cast(),
    ];
    let result = unsafe {
        sys::cuPointerGetAttributes(
            names.len() as u32,
            names.as_mut_ptr(),
            outputs.as_mut_ptr(),
            pointer,
        )
    };
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS);
    attributes
}

#[test]
#[ignore = "requires an Ada GPU with CUDA 13.2 async allocations"]
fn ada_async_allocations_prepare_capture_and_replay_portable_f32() {
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert_eq!(device.compute_capability, (8, 9));
    assert!(
        device.context().has_async_alloc(),
        "Ada regression requires cudarc's async allocation path"
    );
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);

    let dims = (16, 32, 32);
    let a_host = (0..dims.0 * dims.1)
        .map(|index| (index % 17) as f32 * 0.03125 - 0.25)
        .collect::<Vec<_>>();
    let b_host = (0..dims.1 * dims.2)
        .map(|index| (index % 23) as f32 * 0.015625 - 0.125)
        .collect::<Vec<_>>();
    let a = GpuBuffer::from_cpu(&ctx.stream, &a_host).expect("allocate A");
    let b = GpuBuffer::from_cpu(&ctx.stream, &b_host).expect("allocate B");
    let mut output = GpuBuffer::zeros(&ctx.stream, dims.0 * dims.2).expect("allocate output");
    ctx.stream.synchronize().expect("finish async allocations");

    let attributes = pointer_attributes(output.cached_ptr());
    assert!(
        attributes.context.is_null(),
        "CUDA 13.2 async allocation unexpectedly had an associated context: {attributes:?}"
    );
    assert_eq!(attributes.device_ordinal, 0, "{attributes:?}");
    assert_ne!(attributes.buffer_id, 0, "{attributes:?}");
    let offset = output
        .cached_ptr()
        .checked_sub(attributes.range_start)
        .expect("output pointer must not precede its allocation");
    let end = offset
        .checked_add(output.size_bytes() as u64)
        .expect("output range must not overflow");
    assert!(end <= attributes.range_bytes as u64, "{attributes:?}");

    gpu_gemm_bi_forward_raw(&ctx, &mut output, &a, b.cached_ptr(), None, dims)
        .expect("warm production F32 launch with a contextless async allocation");
    ctx.stream
        .synchronize()
        .expect("baseline eager synchronization");
    let eager_bits = output
        .to_cpu(&ctx.stream)
        .expect("download baseline eager output")
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();

    for iteration in 0..3 {
        gpu_gemm_bi_forward_raw(&ctx, &mut output, &a, b.cached_ptr(), None, dims)
            .expect("eager production F32 launch");
        ctx.stream.synchronize().expect("eager synchronization");
        let actual = output
            .to_cpu(&ctx.stream)
            .expect("download eager output")
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, eager_bits, "repeated eager launch {iteration}");
    }

    output
        .zero(&ctx.stream)
        .expect("clear output before capture");
    let graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            gpu_gemm_bi_forward_raw(&ctx, &mut output, &a, b.cached_ptr(), None, dims)
        })
    }
    .expect("capture production F32 launch");
    for iteration in 0..3 {
        graph.launch().expect("replay portable TF32 graph");
        ctx.stream.synchronize().expect("replay synchronization");
        let replay_bits = output
            .to_cpu(&ctx.stream)
            .expect("download replay output")
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(replay_bits, eager_bits, "graph replay {iteration}");
    }
}
