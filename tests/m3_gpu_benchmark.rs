#![cfg(feature = "cuda")]

use std::time::Instant;

use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::inference::GpuMamba3Backbone;
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
use mamba_rs::mamba3_siso::gpu::mamba3_gpu::{
    GpuMamba3BackboneActs, GpuMamba3Dims, GpuMamba3Scratch, GpuMamba3StateBufs, M3Exec,
    gpu_backward_mamba3_backbone, gpu_forward_mamba3_backbone,
};
use mamba_rs::mamba3_siso::gpu::weights::{GpuMamba3Grads, GpuMamba3Weights};
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

/// Full Mamba-3 SISO GPU benchmark: inference + training fwd/bwd.
///
/// Run: `cargo test --features cuda --release --test m3_gpu_benchmark -- --ignored --nocapture`
#[test]
#[ignore]
fn m3_gpu_benchmark() {
    let cfg = Mamba3Config::default();
    let input_dim = cfg.d_model;
    let seq_len = 32;

    let cpu_weights = Mamba3Weights::init(&cfg, input_dim, 42);

    let device = GpuDevice::new(0).unwrap();
    unsafe { device.context().disable_event_tracking() };
    let arch = GpuDevice::nvrtc_arch(device.compute_capability);

    println!();
    println!("=============================================================");
    println!("mamba-3 SISO — Full GPU Benchmark");
    println!("=============================================================");
    println!(
        "Config: d_model={}, layers={}, nheads={}, headdim={}, d_state={}, d_inner={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.nheads(),
        cfg.headdim,
        cfg.d_state,
        cfg.d_inner(),
    );
    println!();

    // ===================================================================
    // 1. GPU Inference (T=1 step, no CUDA Graph)
    // ===================================================================
    println!("--- GPU Inference (T=1, no graph) ---");
    for &b in &[1usize, 4, 16, 64] {
        let mut bb = GpuMamba3Backbone::new(0, &cpu_weights, cfg.clone(), input_dim, b).unwrap();
        let input = vec![0.1f32; b * input_dim];
        let mut output = vec![0.0f32; b * cfg.d_model];

        for _ in 0..20 {
            bb.step(&input, &mut output).unwrap();
        }

        let iters = if b <= 4 { 5000 } else { 2000 };
        let t0 = Instant::now();
        for _ in 0..iters {
            bb.step(&input, &mut output).unwrap();
        }
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        let steps_per_sec = 1_000_000.0 / us;
        println!("  B={b:>4}: {us:>7.1} us/step  ({steps_per_sec:>10.0} steps/sec)");
    }
    println!();

    // ===================================================================
    // 2. GPU Inference (T=1, CUDA Graph)
    // ===================================================================
    println!("--- GPU Inference (T=1, CUDA Graph) ---");
    for &b in &[1usize, 4, 16, 64] {
        let mut bb = GpuMamba3Backbone::new(0, &cpu_weights, cfg.clone(), input_dim, b).unwrap();
        bb.capture_graph().unwrap();
        let input = vec![0.1f32; b * input_dim];
        let mut output = vec![0.0f32; b * cfg.d_model];

        for _ in 0..20 {
            bb.step(&input, &mut output).unwrap();
        }

        let iters = if b <= 4 { 10000 } else { 5000 };
        let t0 = Instant::now();
        for _ in 0..iters {
            bb.step(&input, &mut output).unwrap();
        }
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        let steps_per_sec = 1_000_000.0 / us;
        println!("  B={b:>4}: {us:>7.1} us/step  ({steps_per_sec:>10.0} steps/sec)");
    }
    println!();

    // ===================================================================
    // 3. GPU Training Forward + Backward (B=1, T=seq_len)
    // ===================================================================
    let batch = 1;
    println!("--- GPU Training (B={batch}, T={seq_len}) ---");
    {
        let ctx = GpuCtx::new(&device).unwrap();
        let m3k = Mamba3Kernels::compile(device.context(), arch).unwrap();

        let dims = GpuMamba3Dims {
            batch,
            d_model: cfg.d_model,
            d_inner: cfg.d_inner(),
            d_state: cfg.d_state,
            nheads: cfg.nheads(),
            headdim: cfg.headdim,
            ngroups: cfg.ngroups,
            in_proj_dim: cfg.in_proj_out_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers: cfg.n_layers,
            n_angles: cfg.num_rope_angles(),
            a_floor: cfg.a_floor,
            is_outproj_norm: cfg.is_outproj_norm,
            use_parallel_scan: false,
        };

        let bt = batch * seq_len;
        let nh = cfg.nheads();
        let hd = cfg.headdim;
        let ds = cfg.d_state;
        let na = cfg.num_rope_angles().max(1);
        let nl = cfg.n_layers;

        let gpu_w = GpuMamba3Weights::from_cpu(&ctx.stream, &cpu_weights, &cfg, input_dim).unwrap();
        let input_data: Vec<f32> = (0..bt * input_dim).map(|i| (i as f32) * 0.001).collect();
        let input_gpu = GpuBuffer::from_cpu(&ctx.stream, &input_data).unwrap();

        // Warmup
        for _ in 0..10 {
            let mut temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model).unwrap();
            let mut acts = GpuMamba3BackboneActs::new(&ctx.stream, &dims).unwrap();
            let mut scratch = GpuMamba3Scratch::new(&ctx.stream, &dims).unwrap();
            let mut ssm = GpuBuffer::zeros(&ctx.stream, nl * nh * hd * ds).unwrap();
            let mut k_st = GpuBuffer::zeros(&ctx.stream, nl * nh * ds).unwrap();
            let mut v_st = GpuBuffer::zeros(&ctx.stream, nl * nh * hd).unwrap();
            let mut a_st = GpuBuffer::zeros(&ctx.stream, nl * nh * na).unwrap();
            gpu_forward_mamba3_backbone(
                &M3Exec {
                    ctx: &ctx,
                    kernels: &m3k,
                    dims: &dims,
                },
                &mut temporal,
                &mut acts,
                &gpu_w,
                &input_gpu,
                GpuMamba3StateBufs {
                    ssm: &mut ssm,
                    k: &mut k_st,
                    v: &mut v_st,
                    angle: &mut a_st,
                },
                &mut scratch,
            )
            .unwrap();
        }
        ctx.stream.synchronize().unwrap();

        // Forward only
        let iters = 500;
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model).unwrap();
            let mut acts = GpuMamba3BackboneActs::new(&ctx.stream, &dims).unwrap();
            let mut scratch = GpuMamba3Scratch::new(&ctx.stream, &dims).unwrap();
            let mut ssm = GpuBuffer::zeros(&ctx.stream, nl * nh * hd * ds).unwrap();
            let mut k_st = GpuBuffer::zeros(&ctx.stream, nl * nh * ds).unwrap();
            let mut v_st = GpuBuffer::zeros(&ctx.stream, nl * nh * hd).unwrap();
            let mut a_st = GpuBuffer::zeros(&ctx.stream, nl * nh * na).unwrap();
            gpu_forward_mamba3_backbone(
                &M3Exec {
                    ctx: &ctx,
                    kernels: &m3k,
                    dims: &dims,
                },
                &mut temporal,
                &mut acts,
                &gpu_w,
                &input_gpu,
                GpuMamba3StateBufs {
                    ssm: &mut ssm,
                    k: &mut k_st,
                    v: &mut v_st,
                    angle: &mut a_st,
                },
                &mut scratch,
            )
            .unwrap();
        }
        ctx.stream.synchronize().unwrap();
        let fwd_us = t0.elapsed().as_micros() as f64 / iters as f64;
        println!("  Forward:  {fwd_us:>7.1} us/call");

        // Forward + Backward
        let iters = 200;
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model).unwrap();
            let mut acts = GpuMamba3BackboneActs::new(&ctx.stream, &dims).unwrap();
            let mut scratch = GpuMamba3Scratch::new(&ctx.stream, &dims).unwrap();
            let mut ssm = GpuBuffer::zeros(&ctx.stream, nl * nh * hd * ds).unwrap();
            let mut k_st = GpuBuffer::zeros(&ctx.stream, nl * nh * ds).unwrap();
            let mut v_st = GpuBuffer::zeros(&ctx.stream, nl * nh * hd).unwrap();
            let mut a_st = GpuBuffer::zeros(&ctx.stream, nl * nh * na).unwrap();
            gpu_forward_mamba3_backbone(
                &M3Exec {
                    ctx: &ctx,
                    kernels: &m3k,
                    dims: &dims,
                },
                &mut temporal,
                &mut acts,
                &gpu_w,
                &input_gpu,
                GpuMamba3StateBufs {
                    ssm: &mut ssm,
                    k: &mut k_st,
                    v: &mut v_st,
                    angle: &mut a_st,
                },
                &mut scratch,
            )
            .unwrap();

            let grads = GpuMamba3Grads::new(&ctx.stream, &cfg, input_dim).unwrap();
            let mut d_temporal =
                GpuBuffer::from_cpu(&ctx.stream, &vec![1.0f32; bt * cfg.d_model]).unwrap();
            gpu_backward_mamba3_backbone(
                &M3Exec {
                    ctx: &ctx,
                    kernels: &m3k,
                    dims: &dims,
                },
                &mut d_temporal,
                &acts,
                &gpu_w,
                &grads,
                &mut scratch,
            )
            .unwrap();
        }
        ctx.stream.synchronize().unwrap();
        let fwdbwd_us = t0.elapsed().as_micros() as f64 / iters as f64;
        let bwd_us = fwdbwd_us - fwd_us;
        println!("  Backward: {bwd_us:>7.1} us/call");
        println!("  Fwd+Bwd:  {fwdbwd_us:>7.1} us/call");
    }
    println!();
    println!("=============================================================");
}
