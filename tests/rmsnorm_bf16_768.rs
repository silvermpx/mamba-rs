//! Regression: rmsnorm_fwd_f32in_bf16 kernel must produce non-zero output at
//! d_model=768 (the mamba-130m-hf d_model, only HF size that is not a
//! power-of-2 multiple of the warp size). A previous race between async
//! arena zero-memset and sync weight upload silently zeroed the layer-0
//! norm_weight on this path; this standalone test exercises the kernel with
//! a fresh scale tensor so the bug surfaces here as all-zero output.

#![cfg(feature = "cuda")]

#[test]
fn rmsnorm_bf16_dim768_direct() {
    use cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};
    use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;

    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let stream = ctx.stream.clone();

    let dim: usize = 768;
    let batch: usize = 1;
    let eps: f32 = 1e-5;

    let x_host: Vec<f32> = (0..dim).map(|i| ((i as f32) - 400.0) / 100.0).collect();
    let scale_host: Vec<f32> = vec![1.0; dim];

    let x_dev = stream.clone_htod(&x_host).unwrap();
    let scale_dev = stream.clone_htod(&scale_host).unwrap();
    let y_dev = stream.alloc_zeros::<u16>(dim).unwrap(); // bf16 = u16
    let rms_dev = stream.alloc_zeros::<f32>(batch).unwrap();

    let kernel = ctx
        .kernels
        .rmsnorm_fwd_f32in_typed
        .get(mamba_rs::mamba_ssm::gpu::dtype::WeightDtype::Bf16);

    let block: u32 = (dim as u32).min(1024).next_power_of_two();
    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 4,
    };

    let mut bld = stream.launch_builder(kernel);
    let (y_ptr, _y_g) = y_dev.device_ptr(&stream);
    let (rms_ptr, _r_g) = rms_dev.device_ptr(&stream);
    let (x_ptr, _x_g) = x_dev.device_ptr(&stream);
    let (scale_ptr, _s_g) = scale_dev.device_ptr(&stream);
    bld.arg(&y_ptr);
    bld.arg(&rms_ptr);
    bld.arg(&x_ptr);
    bld.arg(&scale_ptr);
    let b_i = batch as i32;
    let d_i = dim as i32;
    bld.arg(&b_i);
    bld.arg(&d_i);
    bld.arg(&eps);
    unsafe { bld.launch(cfg) }.unwrap();
    stream.synchronize().unwrap();

    let y_host: Vec<u16> = stream.clone_dtoh(&y_dev).unwrap();
    let rms_host: Vec<f32> = stream.clone_dtoh(&rms_dev).unwrap();

    let y_f32: Vec<f32> = y_host
        .iter()
        .map(|&u| half::bf16::from_bits(u).to_f32())
        .collect();
    let nonzero = y_f32.iter().filter(|&&v| v != 0.0).count();

    // Compute expected rms on CPU and compare.
    let ss: f32 = x_host.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let expected_rms = (ss + eps).sqrt();
    let rms_err = (rms_host[0] - expected_rms).abs() / expected_rms;
    assert!(
        rms_err < 1e-4,
        "rms mismatch: got {}, expected {}",
        rms_host[0],
        expected_rms
    );
    assert!(
        nonzero >= dim - 1,
        "bf16 output has {nonzero}/{dim} non-zero elements — expected nearly all"
    );
}
