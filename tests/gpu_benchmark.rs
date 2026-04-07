#![cfg(feature = "cuda")]

use std::time::Instant;

use mamba_rs::MambaBackbone;
use mamba_rs::config::MambaConfig;
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::forward::{
    GpuMambaBackboneActs, GpuMambaDims, GpuMambaScratch, GpuRecurrentState,
    gpu_forward_mamba_backbone,
};
use mamba_rs::gpu::inference::GpuMambaBackbone;
use mamba_rs::gpu::weights::{GpuMambaGrads, GpuMambaTrainWeights};
use mamba_rs::weights::MambaWeights;

/// Full benchmark: GPU inference, GPU inference + CUDA Graph, GPU training fwd/bwd, CPU inference.
///
/// Run: `cargo test --features cuda --test gpu_benchmark -- --ignored --nocapture`
#[test]
#[ignore]
fn full_benchmark() {
    let cfg = MambaConfig::default();
    let input_dim = cfg.d_model;
    let batch_sizes = [1, 4, 16, 64, 128];
    let seq_len = 32;

    let cpu_weights = MambaWeights::init(&cfg, input_dim, 42);

    println!();
    println!("=============================================================");
    println!("mamba-rs v0.1.2 — Full Benchmark");
    println!("=============================================================");
    println!(
        "Config: d_model={}, layers={}, d_inner={}, d_state={}, params={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.d_inner(),
        cfg.d_state,
        MambaBackbone::init(cfg, input_dim, 42).param_count()
    );
    println!();

    // ===================================================================
    // 1. GPU Inference (T=1 step, no CUDA Graph)
    // ===================================================================
    println!("--- GPU Inference (T=1, no graph) ---");
    for &b in &batch_sizes {
        let mut bb = GpuMambaBackbone::new(0, &cpu_weights, cfg, input_dim, b).unwrap();
        let input = vec![0.1f32; b * input_dim];
        let mut output = vec![0.0f32; b * cfg.d_model];

        // Warmup
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
    // 2. GPU Inference + CUDA Graph
    // ===================================================================
    println!("--- GPU Inference (T=1, CUDA Graph) ---");
    for &b in &batch_sizes {
        let mut bb = GpuMambaBackbone::new(0, &cpu_weights, cfg, input_dim, b).unwrap();
        bb.capture_graph().unwrap();
        let input = vec![0.1f32; b * input_dim];
        let mut output = vec![0.0f32; b * cfg.d_model];

        // Warmup with graph
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
    // 3. GPU Training Forward (B=1, T=seq_len)
    // ===================================================================
    println!("--- GPU Training Forward (B=1, T={seq_len}) ---");
    {
        let device = GpuDevice::new(0).unwrap();
        unsafe { device.context().disable_event_tracking() };
        let ctx = GpuCtx::new(&device).unwrap();
        let di = cfg.d_inner();
        let ds = cfg.d_state;

        let gpu_dims = GpuMambaDims {
            batch: 1,
            d_model: cfg.d_model,
            d_inner: di,
            d_state: ds,
            d_conv: cfg.d_conv,
            dt_rank: cfg.dt_rank(),
            xdbl_dim: cfg.xdbl_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers: cfg.n_layers,
        };

        let gpu_w = GpuMambaTrainWeights::from_cpu(&ctx.stream, &cpu_weights).unwrap();
        let input_data: Vec<f32> = (0..seq_len * input_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let input_gpu = GpuBuffer::from_cpu(&ctx.stream, &input_data).unwrap();

        let mut a_neg_cpu = vec![0.0f32; cfg.n_layers * di * ds];
        for (l, lw) in cpu_weights.layers.iter().enumerate() {
            for i in 0..di * ds {
                a_neg_cpu[l * di * ds + i] = -lw.a_log[i].exp();
            }
        }

        // Warmup
        for _ in 0..10 {
            let mut temporal = GpuBuffer::zeros(&ctx.stream, seq_len * cfg.d_model).unwrap();
            let mut acts = GpuMambaBackboneActs::new(&ctx.stream, &gpu_dims).unwrap();
            let mut state = GpuRecurrentState {
                conv_states: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * cfg.d_conv).unwrap(),
                ssm_states: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * ds).unwrap(),
                a_neg_all: GpuBuffer::from_cpu(&ctx.stream, &a_neg_cpu).unwrap(),
            };
            let mut scratch = GpuMambaScratch::new(&ctx.stream, &gpu_dims).unwrap();
            gpu_forward_mamba_backbone(
                &ctx,
                &mut temporal,
                &mut acts,
                &gpu_w,
                &input_gpu,
                &mut state,
                &mut scratch,
            )
            .unwrap();
        }
        ctx.stream.synchronize().unwrap();

        // Bench
        let iters = 500;
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut temporal = GpuBuffer::zeros(&ctx.stream, seq_len * cfg.d_model).unwrap();
            let mut acts = GpuMambaBackboneActs::new(&ctx.stream, &gpu_dims).unwrap();
            let mut state = GpuRecurrentState {
                conv_states: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * cfg.d_conv).unwrap(),
                ssm_states: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * ds).unwrap(),
                a_neg_all: GpuBuffer::from_cpu(&ctx.stream, &a_neg_cpu).unwrap(),
            };
            let mut scratch = GpuMambaScratch::new(&ctx.stream, &gpu_dims).unwrap();
            gpu_forward_mamba_backbone(
                &ctx,
                &mut temporal,
                &mut acts,
                &gpu_w,
                &input_gpu,
                &mut state,
                &mut scratch,
            )
            .unwrap();
        }
        ctx.stream.synchronize().unwrap();
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        println!("  Forward:  {us:>7.1} us/call");

        // Forward + Backward
        let iters = 200;
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut temporal = GpuBuffer::zeros(&ctx.stream, seq_len * cfg.d_model).unwrap();
            let mut acts = GpuMambaBackboneActs::new(&ctx.stream, &gpu_dims).unwrap();
            let mut state = GpuRecurrentState {
                conv_states: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * cfg.d_conv).unwrap(),
                ssm_states: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * ds).unwrap(),
                a_neg_all: GpuBuffer::from_cpu(&ctx.stream, &a_neg_cpu).unwrap(),
            };
            let mut scratch = GpuMambaScratch::new(&ctx.stream, &gpu_dims).unwrap();
            gpu_forward_mamba_backbone(
                &ctx,
                &mut temporal,
                &mut acts,
                &gpu_w,
                &input_gpu,
                &mut state,
                &mut scratch,
            )
            .unwrap();
            let grads = GpuMambaGrads::new(&ctx.stream, &cfg, input_dim).unwrap();
            let mut d_temporal =
                GpuBuffer::from_cpu(&ctx.stream, &vec![1.0f32; seq_len * cfg.d_model]).unwrap();
            mamba_rs::gpu::backward::gpu_backward_mamba_backbone(
                &ctx,
                &mut d_temporal,
                &grads,
                &acts,
                &gpu_w,
                &state.a_neg_all,
                &mut scratch,
            )
            .unwrap();
        }
        ctx.stream.synchronize().unwrap();
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        println!("  Fwd+Bwd:  {us:>7.1} us/call");
    }
    println!();

    // ===================================================================
    // 4. CPU Inference (T=1 step)
    // ===================================================================
    println!("--- CPU Inference (T=1) ---");
    {
        let bb = MambaBackbone::init(cfg, input_dim, 42);
        let mut state = bb.alloc_state();
        let mut scratch = bb.alloc_scratch();
        let mut output = vec![0.0f32; cfg.d_model];
        let input = vec![0.1f32; input_dim];

        // Warmup
        for _ in 0..100 {
            bb.forward_step(&input, &mut output, &mut state, &mut scratch);
        }
        state.reset();

        let iters = 10_000;
        let t0 = Instant::now();
        for _ in 0..iters {
            bb.forward_step(&input, &mut output, &mut state, &mut scratch);
        }
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        let steps_per_sec = 1_000_000.0 / us;
        println!("  B=1: {us:>7.1} us/step  ({steps_per_sec:>10.0} steps/sec)");
    }
    println!();

    // ===================================================================
    // 5. CPU Training Forward+Backward (B=1, T=seq_len)
    // ===================================================================
    println!("--- CPU Training (B=1, T={seq_len}) ---");
    {
        use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
        use mamba_rs::train::flat::MambaBackboneFlat;
        use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
        use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};

        let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
        let di = dims.d_inner;
        let ds = dims.d_state;

        let tw = TrainMambaWeights {
            input_proj_w: cpu_weights.input_proj_w.clone(),
            input_proj_b: cpu_weights.input_proj_b.clone(),
            layers: cpu_weights
                .layers
                .iter()
                .map(|lw| TrainMambaLayerWeights {
                    norm_weight: lw.norm_weight.clone(),
                    in_proj_w: lw.in_proj_w.clone(),
                    conv1d_weight: lw.conv1d_weight.clone(),
                    conv1d_bias: lw.conv1d_bias.clone(),
                    x_proj_w: lw.x_proj_w.clone(),
                    dt_proj_w: lw.dt_proj_w.clone(),
                    dt_proj_b: lw.dt_proj_b.clone(),
                    a_log: lw.a_log.clone(),
                    d_param: lw.d_param.clone(),
                    out_proj_w: lw.out_proj_w.clone(),
                })
                .collect(),
            norm_f_weight: cpu_weights.norm_f_weight.clone(),
        };

        let mut a_neg = vec![0.0f32; cfg.n_layers * di * ds];
        for (l, lw) in tw.layers.iter().enumerate() {
            for i in 0..di * ds {
                a_neg[l * di * ds + i] = -lw.a_log[i].exp();
            }
        }

        let input_data: Vec<f32> = (0..seq_len * input_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();

        // Forward only
        let iters = 500;
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut acts = MambaBackboneFlat::zeros(dims);
            let mut scratch = PhaseScratch::zeros(&dims);
            let mut conv = vec![0.0f32; cfg.n_layers * di * cfg.d_conv];
            let mut ssm = vec![0.0f32; cfg.n_layers * di * ds];
            let mut state = MambaRecurrentState {
                conv: &mut conv,
                ssm: &mut ssm,
                a_neg: &a_neg,
            };
            let mut temporal = vec![0.0f32; seq_len * cfg.d_model];
            mamba_rs::train::forward::forward_mamba_backbone_batched(
                &mut temporal,
                &mut acts,
                &tw,
                &input_data,
                &mut state,
                &mut scratch,
                &dims,
            );
        }
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        println!("  Forward:  {us:>7.1} us/call");

        // Forward + Backward
        let iters = 200;
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut acts = MambaBackboneFlat::zeros(dims);
            let mut scratch = PhaseScratch::zeros(&dims);
            let mut conv = vec![0.0f32; cfg.n_layers * di * cfg.d_conv];
            let mut ssm = vec![0.0f32; cfg.n_layers * di * ds];
            let mut state = MambaRecurrentState {
                conv: &mut conv,
                ssm: &mut ssm,
                a_neg: &a_neg,
            };
            let mut temporal = vec![0.0f32; seq_len * cfg.d_model];
            mamba_rs::train::forward::forward_mamba_backbone_batched(
                &mut temporal,
                &mut acts,
                &tw,
                &input_data,
                &mut state,
                &mut scratch,
                &dims,
            );
            let mut d_temporal = vec![1.0f32; seq_len * cfg.d_model];
            let mut grad_tw = TrainMambaWeights::zeros_from_dims(&dims);
            let mut bwd_scratch = BackwardPhaseScratch::zeros(&dims);
            mamba_rs::train::backward::backward_mamba_backbone_batched(
                &mut d_temporal,
                &mut grad_tw,
                &acts,
                &tw,
                &a_neg,
                &mut bwd_scratch,
                &dims,
            );
        }
        let us = t0.elapsed().as_micros() as f64 / iters as f64;
        println!("  Fwd+Bwd:  {us:>7.1} us/call");
    }
    println!();
    println!("=============================================================");
}
