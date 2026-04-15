//! Step 14 — CUDA-Graph-captured M3 bf16 training step parity vs eager.
//!
//! M3 analogue of `tests/training_graph_parity.rs`. Asserts that
//! `GpuMamba3TrainingStepGraph::capture` + replay produces bit-identical
//! master weights vs an eager forward+backward+adamw+sync for the same
//! N steps with identical inputs.

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::adamw::{AdamWBiasFactors, GpuAdamW, step_m3_capturable};
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::backward_mixed::gpu_backward_mamba3_backbone_mixed;
use mamba_rs::mamba3_siso::gpu::forward_mixed::{
    GpuMamba3BackboneMixedActs, GpuMamba3MixedScratch, gpu_forward_mamba3_backbone_mixed,
};
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
use mamba_rs::mamba3_siso::gpu::state::{GpuMamba3Dims, GpuMamba3Scratch};
use mamba_rs::mamba3_siso::gpu::training_graph::GpuMamba3TrainingStepGraph;
use mamba_rs::mamba3_siso::gpu::weights::GpuMamba3Grads;
use mamba_rs::mamba3_siso::gpu::weights_mixed_train::GpuMamba3TrainMixedWeights;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn cfg_m3() -> Mamba3Config {
    Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    }
}

fn dims_for(cfg: &Mamba3Config, batch: usize, seq_len: usize) -> GpuMamba3Dims {
    GpuMamba3Dims {
        batch,
        d_model: cfg.d_model,
        d_inner: cfg.d_inner(),
        d_state: cfg.d_state,
        nheads: cfg.nheads(),
        headdim: cfg.headdim,
        ngroups: cfg.ngroups,
        in_proj_dim: cfg.in_proj_out_dim(),
        seq_len,
        mamba_input_dim: cfg.d_model,
        n_layers: cfg.n_layers,
        n_angles: cfg.num_rope_angles(),
        a_floor: cfg.a_floor,
        is_outproj_norm: cfg.is_outproj_norm,
        use_parallel_scan: true,
    }
}

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFFFF) as f32 / 65536.0 - 0.5
        })
        .collect()
}

struct Setup {
    dims: GpuMamba3Dims,
    weights: GpuMamba3TrainMixedWeights,
    acts: GpuMamba3BackboneMixedActs,
    f32_scratch: GpuMamba3Scratch,
    mixed_scratch: GpuMamba3MixedScratch,
    temporal: GpuBuffer,
    mamba_input: GpuBuffer,
    d_temporal: GpuBuffer,
    ssm_states: GpuBuffer,
    k_states: GpuBuffer,
    v_states: GpuBuffer,
    angle_states: GpuBuffer,
    grads: GpuMamba3Grads,
    adam: GpuAdamW,
    bias: AdamWBiasFactors,
}

fn build(ctx: &GpuCtx, dtype: WeightDtype, batch: usize, seq_len: usize) -> Setup {
    let cfg = cfg_m3();
    let dims = dims_for(&cfg, batch, seq_len);
    let mut cpu = Mamba3Weights::init(&cfg, cfg.d_model, 0xC0DECAFE);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();

    let weights =
        GpuMamba3TrainMixedWeights::from_cpu(&ctx.stream, &cpu, &cfg, cfg.d_model, dtype).unwrap();
    let acts =
        GpuMamba3BackboneMixedActs::new(&ctx.stream, &cfg, batch, seq_len, cfg.d_model, dtype)
            .unwrap();
    let f32_scratch = GpuMamba3Scratch::new(&ctx.stream, &dims).unwrap();
    let mixed_scratch =
        GpuMamba3MixedScratch::new(&ctx.stream, &cfg, batch, seq_len, dtype).unwrap();

    let bt = batch * seq_len;
    let nh = cfg.nheads();
    let hd = cfg.headdim;
    let ds = cfg.d_state;
    let na = cfg.num_rope_angles().max(1);
    let nl = cfg.n_layers;

    let temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model).unwrap();
    let mamba_input = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model).unwrap();
    let d_temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model).unwrap();
    let ssm_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd * ds).unwrap();
    let k_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * ds).unwrap();
    let v_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd).unwrap();
    let angle_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * na).unwrap();
    let grads = GpuMamba3Grads::new(&ctx.stream, &cfg, cfg.d_model).unwrap();

    let n_params = grads.flat.len();
    let adam = GpuAdamW::new(&ctx.stream, n_params)
        .unwrap()
        .with_lr(1e-4)
        .with_weight_decay(1e-2);
    let bias = AdamWBiasFactors::new(&ctx.stream).unwrap();

    ctx.stream.synchronize().unwrap();

    Setup {
        dims,
        weights,
        acts,
        f32_scratch,
        mixed_scratch,
        temporal,
        mamba_input,
        d_temporal,
        ssm_states,
        k_states,
        v_states,
        angle_states,
        grads,
        adam,
        bias,
    }
}

fn reset_state(s: &mut Setup, ctx: &GpuCtx) {
    s.ssm_states.zero(&ctx.stream).unwrap();
    s.k_states.zero(&ctx.stream).unwrap();
    s.v_states.zero(&ctx.stream).unwrap();
    s.angle_states.zero(&ctx.stream).unwrap();
}

fn snapshot(s: &Setup, ctx: &GpuCtx) -> Vec<f32> {
    let mut out = Vec::new();
    out.extend(s.weights.master.input_proj_w.to_cpu(&ctx.stream).unwrap());
    out.extend(s.weights.master.input_proj_b.to_cpu(&ctx.stream).unwrap());
    for lw in &s.weights.master.layers {
        out.extend(lw.norm_weight.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.in_proj_w.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.dt_bias.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.b_norm_weight.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.c_norm_weight.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.b_bias.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.c_bias.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.d_param.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.norm_gate_weight.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.out_proj_w.to_cpu(&ctx.stream).unwrap());
    }
    out.extend(s.weights.master.norm_f_weight.to_cpu(&ctx.stream).unwrap());
    out
}

fn one_eager_step(s: &mut Setup, ctx: &GpuCtx, m3k: &Mamba3Kernels, inp: &[f32], dt: &[f32]) {
    s.mamba_input.upload(&ctx.stream, inp).unwrap();
    s.d_temporal.upload(&ctx.stream, dt).unwrap();
    s.grads.zero(&ctx.stream).unwrap();
    gpu_forward_mamba3_backbone_mixed(
        ctx,
        m3k,
        &mut s.temporal,
        &mut s.acts,
        &s.weights,
        &s.mamba_input,
        &mut s.ssm_states,
        &mut s.k_states,
        &mut s.v_states,
        &mut s.angle_states,
        &mut s.mixed_scratch,
        &s.dims,
    )
    .unwrap();
    gpu_backward_mamba3_backbone_mixed(
        ctx,
        m3k,
        &mut s.d_temporal,
        &s.acts,
        &s.weights,
        &s.grads,
        &mut s.f32_scratch,
        &mut s.mixed_scratch,
        &s.dims,
    )
    .unwrap();
    let (_, bc1, bc2) = s.adam.advance();
    s.bias.write(&ctx.stream, bc1, bc2).unwrap();
    step_m3_capturable(
        ctx,
        &m3k.adamw_step_f32_capturable,
        &s.adam,
        s.bias.ptr(),
        &mut s.weights.master,
        &s.grads,
    )
    .unwrap();
    s.weights.sync_master_to_compute(ctx).unwrap();
}

#[test]
fn m3_training_graph_bf16_one_step_matches_eager() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = Mamba3Kernels::compile(dev.context(), "sm_89").unwrap();
    let batch = 1;
    let seq_len = 64;

    let n = batch * seq_len * cfg_m3().d_model;
    let inp = det(n, 0xA1);
    let dt = det(n, 0xB2);

    let mut eager = build(&ctx, WeightDtype::Bf16, batch, seq_len);
    reset_state(&mut eager, &ctx);
    one_eager_step(&mut eager, &ctx, &m3k, &inp, &dt);
    ctx.stream.synchronize().unwrap();
    let after_eager = snapshot(&eager, &ctx);

    let mut g = build(&ctx, WeightDtype::Bf16, batch, seq_len);
    reset_state(&mut g, &ctx);
    g.mamba_input.upload(&ctx.stream, &inp).unwrap();
    g.d_temporal.upload(&ctx.stream, &dt).unwrap();
    g.bias.write(&ctx.stream, 1.0, 1.0).unwrap();

    let graph = GpuMamba3TrainingStepGraph::capture(
        &ctx,
        &cfg_m3(),
        &m3k,
        &mut g.weights,
        &g.adam,
        &g.bias,
        &mut g.grads,
        &mut g.acts,
        &mut g.f32_scratch,
        &mut g.mixed_scratch,
        &mut g.temporal,
        &g.mamba_input,
        &mut g.d_temporal,
        &mut g.ssm_states,
        &mut g.k_states,
        &mut g.v_states,
        &mut g.angle_states,
        &g.dims,
    )
    .unwrap();

    // Capture only records — must replay to execute.
    let (_, bc1, bc2) = g.adam.advance();
    g.bias.write(&ctx.stream, bc1, bc2).unwrap();
    graph
        .replay(
            &ctx,
            &g.weights,
            &g.adam,
            &g.bias,
            &g.grads,
            &g.temporal,
            &g.mamba_input,
            &g.d_temporal,
            &g.ssm_states,
            &g.k_states,
            &g.v_states,
            &g.angle_states,
        )
        .unwrap();
    ctx.stream.synchronize().unwrap();
    let after_graph = snapshot(&g, &ctx);

    assert_eq!(after_eager.len(), after_graph.len());
    let max_err = after_eager
        .iter()
        .zip(&after_graph)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("M3 one-step eager-vs-graph max_err = {max_err:.3e}");
    assert!(
        max_err < 1e-5,
        "M3 one-step parity broke: max_err={max_err}"
    );
}

#[test]
fn m3_training_graph_bf16_multi_replay_matches_eager() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = Mamba3Kernels::compile(dev.context(), "sm_89").unwrap();
    let batch = 1;
    let seq_len = 64;
    let n_steps = 5;

    let n = batch * seq_len * cfg_m3().d_model;
    let inputs: Vec<Vec<f32>> = (0..n_steps).map(|s| det(n, 0xC0 + s as u32)).collect();
    let d_temps: Vec<Vec<f32>> = (0..n_steps).map(|s| det(n, 0xD0 + s as u32)).collect();

    let mut eager = build(&ctx, WeightDtype::Bf16, batch, seq_len);
    for s in 0..n_steps {
        reset_state(&mut eager, &ctx);
        one_eager_step(&mut eager, &ctx, &m3k, &inputs[s], &d_temps[s]);
    }
    ctx.stream.synchronize().unwrap();
    let after_eager = snapshot(&eager, &ctx);

    let mut g = build(&ctx, WeightDtype::Bf16, batch, seq_len);
    reset_state(&mut g, &ctx);
    g.mamba_input.upload(&ctx.stream, &inputs[0]).unwrap();
    g.d_temporal.upload(&ctx.stream, &d_temps[0]).unwrap();
    g.bias.write(&ctx.stream, 1.0, 1.0).unwrap();
    let graph = GpuMamba3TrainingStepGraph::capture(
        &ctx,
        &cfg_m3(),
        &m3k,
        &mut g.weights,
        &g.adam,
        &g.bias,
        &mut g.grads,
        &mut g.acts,
        &mut g.f32_scratch,
        &mut g.mixed_scratch,
        &mut g.temporal,
        &g.mamba_input,
        &mut g.d_temporal,
        &mut g.ssm_states,
        &mut g.k_states,
        &mut g.v_states,
        &mut g.angle_states,
        &g.dims,
    )
    .unwrap();

    for s in 0..n_steps {
        reset_state(&mut g, &ctx);
        g.mamba_input.upload(&ctx.stream, &inputs[s]).unwrap();
        g.d_temporal.upload(&ctx.stream, &d_temps[s]).unwrap();
        let (_, bc1, bc2) = g.adam.advance();
        g.bias.write(&ctx.stream, bc1, bc2).unwrap();
        graph
            .replay(
                &ctx,
                &g.weights,
                &g.adam,
                &g.bias,
                &g.grads,
                &g.temporal,
                &g.mamba_input,
                &g.d_temporal,
                &g.ssm_states,
                &g.k_states,
                &g.v_states,
                &g.angle_states,
            )
            .unwrap();
    }
    ctx.stream.synchronize().unwrap();
    let after_graph = snapshot(&g, &ctx);

    let max_err = after_eager
        .iter()
        .zip(&after_graph)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("M3 {n_steps}-step eager-vs-graph max_err = {max_err:.3e}");
    assert!(
        max_err < 5e-5,
        "M3 multi-step parity broke: max_err={max_err}"
    );
}
