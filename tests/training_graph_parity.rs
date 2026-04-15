//! Step 14 — CUDA-Graph-captured M1 bf16 training step parity vs eager.
//!
//! Runs N steps eagerly, then N steps via captured-graph replay (with
//! identical inputs/seeds/weights). Asserts final master weights are
//! bit-close. This proves:
//!   - Captured graph replays produce the same numerics as eager
//!   - Per-step bias-factor refresh works (capturable AdamW)
//!   - Master→compute sync is captured correctly
//!   - Pointer-stability invariant holds across many replays

#![cfg(feature = "cuda")]

use mamba_rs::config::MambaConfig;
use mamba_rs::mamba_ssm::gpu::adamw::{AdamWBiasFactors, GpuAdamW, step_m1_capturable};
use mamba_rs::mamba_ssm::gpu::backward_mixed::gpu_backward_mamba_backbone_mixed;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::forward::{GpuMambaDims, GpuRecurrentState};
use mamba_rs::mamba_ssm::gpu::forward_mixed::{
    GpuMambaBackboneMixedActs, GpuMambaMixedTrainScratch, gpu_forward_mamba_backbone_train_mixed,
};
use mamba_rs::mamba_ssm::gpu::training_graph::GpuMambaTrainingStepGraph;
use mamba_rs::mamba_ssm::gpu::weights::GpuMambaGrads;
use mamba_rs::mamba_ssm::gpu::weights_mixed_train::GpuMambaTrainMixedWeights;
use mamba_rs::weights::MambaWeights;

fn tiny_cfg() -> MambaConfig {
    MambaConfig {
        d_model: 32,
        n_layers: 1,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
    }
}

fn det_input(n: usize, seed: u32) -> Vec<f32> {
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
    cfg: MambaConfig,
    weights: GpuMambaTrainMixedWeights,
    acts: GpuMambaBackboneMixedActs,
    scratch: GpuMambaMixedTrainScratch,
    state: GpuRecurrentState,
    a_neg_all: GpuBuffer,
    mamba_input: GpuBuffer,
    d_temporal: GpuBuffer,
    grads: GpuMambaGrads,
    adam: GpuAdamW,
    bias: AdamWBiasFactors,
}

fn build_setup(ctx: &GpuCtx, dtype: WeightDtype, batch: usize, seq_len: usize) -> Setup {
    let cfg = tiny_cfg();
    let d_model = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let d_conv = cfg.d_conv;
    let dt_rank = cfg.dt_rank();
    let xdbl_dim = cfg.xdbl_dim();
    let n_layers = cfg.n_layers;

    let mut cpu = MambaWeights::init(&cfg, d_model, 0xBADF00D);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    let weights = GpuMambaTrainMixedWeights::from_cpu(&ctx.stream, &cpu, &cfg, dtype).unwrap();

    let dims = GpuMambaDims {
        batch,
        d_model,
        d_inner: di,
        d_state: ds,
        d_conv,
        dt_rank,
        xdbl_dim,
        seq_len,
        mamba_input_dim: d_model,
        n_layers,
    };

    let acts = GpuMambaBackboneMixedActs::new(&ctx.stream, &dims, dtype).unwrap();
    let scratch = GpuMambaMixedTrainScratch::new(&ctx.stream, &dims, dtype).unwrap();

    let state = GpuRecurrentState {
        conv_states: GpuBuffer::zeros(&ctx.stream, n_layers * di * d_conv).unwrap(),
        ssm_states: GpuBuffer::zeros(&ctx.stream, n_layers * di * ds).unwrap(),
        a_neg_all: GpuBuffer::zeros(&ctx.stream, n_layers * di * ds).unwrap(),
    };
    let mut a_neg_flat = vec![0.0f32; n_layers * di * ds];
    for (l, lw) in cpu.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg_flat[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }
    let mut a_neg_all = GpuBuffer::zeros(&ctx.stream, n_layers * di * ds).unwrap();
    a_neg_all.upload(&ctx.stream, &a_neg_flat).unwrap();

    let mamba_input = GpuBuffer::zeros(&ctx.stream, batch * seq_len * d_model).unwrap();
    let d_temporal = GpuBuffer::zeros(&ctx.stream, batch * seq_len * d_model).unwrap();
    let grads = GpuMambaGrads::new(&ctx.stream, &cfg, d_model).unwrap();
    let n_params = grads.flat.len();

    let adam = GpuAdamW::new(&ctx.stream, n_params)
        .unwrap()
        .with_lr(1e-4)
        .with_weight_decay(1e-2);
    let bias = AdamWBiasFactors::new(&ctx.stream).unwrap();

    ctx.stream.synchronize().unwrap();

    Setup {
        cfg,
        weights,
        acts,
        scratch,
        state,
        a_neg_all,
        mamba_input,
        d_temporal,
        grads,
        adam,
        bias,
    }
}

fn reset_state(setup: &mut Setup, ctx: &GpuCtx) {
    let n_layers = setup.cfg.n_layers;
    let di = setup.cfg.d_inner();
    let ds = setup.cfg.d_state;
    let d_conv = setup.cfg.d_conv;

    setup.state.conv_states.zero(&ctx.stream).unwrap();
    setup.state.ssm_states.zero(&ctx.stream).unwrap();
    let _ = (n_layers, di, ds, d_conv);
}

fn snapshot_master(s: &Setup, ctx: &GpuCtx) -> Vec<f32> {
    let mut out = Vec::new();
    out.extend(s.weights.master.input_proj_w.to_cpu(&ctx.stream).unwrap());
    out.extend(s.weights.master.input_proj_b.to_cpu(&ctx.stream).unwrap());
    for lw in &s.weights.master.layers {
        out.extend(lw.norm_weight.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.in_proj_w.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.conv1d_weight.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.conv1d_bias.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.x_proj_w.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.dt_proj_w.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.dt_proj_b.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.a_log.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.d_param.to_cpu(&ctx.stream).unwrap());
        out.extend(lw.out_proj_w.to_cpu(&ctx.stream).unwrap());
    }
    out.extend(s.weights.master.norm_f_weight.to_cpu(&ctx.stream).unwrap());
    out
}

fn one_eager_step(setup: &mut Setup, ctx: &GpuCtx, input: &[f32], d_temp: &[f32]) {
    setup.mamba_input.upload(&ctx.stream, input).unwrap();
    setup.d_temporal.upload(&ctx.stream, d_temp).unwrap();
    setup.grads.zero(&ctx.stream).unwrap();
    gpu_forward_mamba_backbone_train_mixed(
        ctx,
        &mut setup.acts,
        &setup.weights,
        &setup.mamba_input,
        &mut setup.state,
        &mut setup.scratch,
    )
    .unwrap();
    gpu_backward_mamba_backbone_mixed(
        ctx,
        &mut setup.d_temporal,
        &setup.grads,
        &setup.acts,
        &setup.weights.compute,
        &setup.a_neg_all,
        &mut setup.scratch,
    )
    .unwrap();
    // Use the capturable kernel + device-buf bias factors in eager mode too,
    // so eager and graph share a SINGLE kernel implementation. Otherwise
    // the two adamw variants drift by ~1e-4 per step (different
    // generated-PTX register pressure on bias factors).
    let (_, bc1, bc2) = setup.adam.advance();
    setup.bias.write(&ctx.stream, bc1, bc2).unwrap();
    step_m1_capturable(
        ctx,
        &ctx.kernels.adamw_step_f32_capturable,
        &setup.adam,
        setup.bias.ptr(),
        &mut setup.weights.master,
        &setup.grads,
    )
    .unwrap();
    setup.weights.sync_master_to_compute(ctx).unwrap();
}

#[test]
fn training_graph_bf16_one_step_matches_eager() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let batch = 1;
    let seq_len = 4;

    let n = batch * seq_len * tiny_cfg().d_model;
    let input = det_input(n, 0xA1);
    let d_temp = det_input(n, 0xB2);

    // EAGER path
    let mut eager = build_setup(&ctx, WeightDtype::Bf16, batch, seq_len);
    reset_state(&mut eager, &ctx);
    one_eager_step(&mut eager, &ctx, &input, &d_temp);
    ctx.stream.synchronize().unwrap();
    let after_eager = snapshot_master(&eager, &ctx);

    // GRAPH path: identical setup, capture, single replay.
    let mut g = build_setup(&ctx, WeightDtype::Bf16, batch, seq_len);
    reset_state(&mut g, &ctx);
    g.mamba_input.upload(&ctx.stream, &input).unwrap();
    g.d_temporal.upload(&ctx.stream, &d_temp).unwrap();

    // Pre-compute step-1 bias factors and write to bias buffer (this is
    // what would normally happen via adam.advance() → bias.write()).
    let (_, bc1, bc2) = g.adam.advance();
    g.bias.write(&ctx.stream, bc1, bc2).unwrap();

    let graph = GpuMambaTrainingStepGraph::capture(
        &ctx,
        &g.cfg,
        &mut g.weights,
        &g.adam,
        &g.bias,
        &mut g.grads,
        &mut g.acts,
        &mut g.scratch,
        &g.a_neg_all,
        &g.mamba_input,
        &mut g.d_temporal,
        &mut g.state,
        batch,
        seq_len,
    )
    .unwrap();
    // cuStreamBeginCapture only RECORDS — must call replay() to execute.
    graph
        .replay(
            &ctx,
            &g.weights,
            &g.adam,
            &g.bias,
            &g.grads,
            &g.a_neg_all,
            &g.mamba_input,
            &g.d_temporal,
            &g.state,
        )
        .unwrap();
    ctx.stream.synchronize().unwrap();
    let after_graph_capture = snapshot_master(&g, &ctx);

    assert_eq!(after_eager.len(), after_graph_capture.len());
    let mut max_err = 0.0f32;
    for (a, b) in after_eager.iter().zip(&after_graph_capture) {
        let e = (a - b).abs();
        if e > max_err {
            max_err = e;
        }
    }
    eprintln!("one-step eager-vs-graph max_err = {max_err:.3e}");
    assert!(max_err < 1e-5, "one-step parity broke: max_err={max_err}");
}

#[test]
fn training_graph_bf16_multi_replay_matches_eager() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let batch = 1;
    let seq_len = 4;
    let n_steps = 5;

    let n = batch * seq_len * tiny_cfg().d_model;
    let inputs: Vec<Vec<f32>> = (0..n_steps)
        .map(|s| det_input(n, 0xC0 + s as u32))
        .collect();
    let d_temps: Vec<Vec<f32>> = (0..n_steps)
        .map(|s| det_input(n, 0xD0 + s as u32))
        .collect();

    // EAGER: N steps from scratch.
    let mut eager = build_setup(&ctx, WeightDtype::Bf16, batch, seq_len);
    for s in 0..n_steps {
        reset_state(&mut eager, &ctx);
        one_eager_step(&mut eager, &ctx, &inputs[s], &d_temps[s]);
    }
    ctx.stream.synchronize().unwrap();
    let after_eager = snapshot_master(&eager, &ctx);

    // GRAPH: capture once (records, doesn't execute), then replay N times.
    let mut g = build_setup(&ctx, WeightDtype::Bf16, batch, seq_len);

    // Capture against step-1 inputs (pointers must be stable; values
    // refreshed per replay).
    reset_state(&mut g, &ctx);
    g.mamba_input.upload(&ctx.stream, &inputs[0]).unwrap();
    g.d_temporal.upload(&ctx.stream, &d_temps[0]).unwrap();
    g.bias.write(&ctx.stream, 1.0, 1.0).unwrap(); // dummy; real values per replay
    let graph = GpuMambaTrainingStepGraph::capture(
        &ctx,
        &g.cfg,
        &mut g.weights,
        &g.adam,
        &g.bias,
        &mut g.grads,
        &mut g.acts,
        &mut g.scratch,
        &g.a_neg_all,
        &g.mamba_input,
        &mut g.d_temporal,
        &mut g.state,
        batch,
        seq_len,
    )
    .unwrap();

    // Replay N times — each replay = one full training step.
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
                &g.a_neg_all,
                &g.mamba_input,
                &g.d_temporal,
                &g.state,
            )
            .unwrap();
    }
    ctx.stream.synchronize().unwrap();
    let after_graph = snapshot_master(&g, &ctx);

    assert_eq!(after_eager.len(), after_graph.len());
    let mut max_err = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq = 0.0f64;
    for (a, b) in after_eager.iter().zip(&after_graph) {
        let e = (a - b).abs();
        if e > max_err {
            max_err = e;
        }
        sum_sq_err += ((a - b) as f64).powi(2);
        sum_sq += (*a as f64).powi(2);
    }
    let rel = (sum_sq_err / sum_sq.max(1e-30)).sqrt();
    eprintln!("{n_steps}-step eager-vs-graph: max_err={max_err:.3e}, rel_l2={rel:.3e}");
    assert!(max_err < 5e-5, "multi-step parity broke: max_err={max_err}");
}

/// Sanity: the pointer-stability invariant fires when a buffer is
/// reallocated between capture and replay.
#[test]
#[should_panic(expected = "pointer changed since capture")]
fn training_graph_panics_on_pointer_mismatch() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let batch = 1;
    let seq_len = 4;
    let n = batch * seq_len * tiny_cfg().d_model;

    let mut g = build_setup(&ctx, WeightDtype::Bf16, batch, seq_len);
    reset_state(&mut g, &ctx);
    g.mamba_input.upload(&ctx.stream, &det_input(n, 1)).unwrap();
    g.d_temporal.upload(&ctx.stream, &det_input(n, 2)).unwrap();
    let (_, bc1, bc2) = g.adam.advance();
    g.bias.write(&ctx.stream, bc1, bc2).unwrap();

    let graph = GpuMambaTrainingStepGraph::capture(
        &ctx,
        &g.cfg,
        &mut g.weights,
        &g.adam,
        &g.bias,
        &mut g.grads,
        &mut g.acts,
        &mut g.scratch,
        &g.a_neg_all,
        &g.mamba_input,
        &mut g.d_temporal,
        &mut g.state,
        batch,
        seq_len,
    )
    .unwrap();

    // Reallocate mamba_input — different cached_ptr, must panic on replay.
    let new_input = GpuBuffer::zeros(&ctx.stream, n).unwrap();
    let (_, bc1, bc2) = g.adam.advance();
    g.bias.write(&ctx.stream, bc1, bc2).unwrap();
    graph
        .replay(
            &ctx,
            &g.weights,
            &g.adam,
            &g.bias,
            &g.grads,
            &g.a_neg_all,
            &new_input, // ← different buffer
            &g.d_temporal,
            &g.state,
        )
        .unwrap();
}
