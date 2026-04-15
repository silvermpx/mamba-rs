//! Step 14 follow-up — f32 training graph parity vs eager (M1 + M3).
//!
//! The bf16-mixed training graphs were validated in
//! `tests/training_graph_parity.rs` and `tests/m3_training_graph_parity.rs`.
//! This test covers the f32 graphs added later: the same forward+backward+
//! adamw pipeline but with no mixed-precision shadow weights and no
//! sync_master_to_compute step.

#![cfg(feature = "cuda")]

use mamba_rs::config::MambaConfig;
use mamba_rs::mamba_ssm::gpu::adamw::{AdamWBiasFactors, GpuAdamW, step_m1_capturable};
use mamba_rs::mamba_ssm::gpu::backward::gpu_backward_mamba_backbone;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::forward::{
    GpuMambaBackboneActs, GpuMambaDims, GpuMambaScratch, GpuRecurrentState,
    gpu_forward_mamba_backbone,
};
use mamba_rs::mamba_ssm::gpu::training_graph::GpuMambaF32TrainingStepGraph;
use mamba_rs::mamba_ssm::gpu::weights::{GpuMambaGrads, GpuMambaTrainWeights};
use mamba_rs::weights::MambaWeights;

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

fn cfg() -> MambaConfig {
    MambaConfig {
        d_model: 32,
        n_layers: 1,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
    }
}

#[test]
fn m1_f32_training_graph_matches_eager() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let cfg = cfg();
    let batch = 1;
    let seq_len = 4;
    let dm = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let dc = cfg.d_conv;
    let dr = cfg.dt_rank();
    let xd = cfg.xdbl_dim();
    let nl = cfg.n_layers;

    // f32 forward has NO identity-proj branch (mixed forward does). Keep
    // input_proj_w populated.
    let mut cpu = MambaWeights::init(&cfg, dm, 0xF0017A);
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    let dims = GpuMambaDims {
        batch,
        d_model: dm,
        d_inner: di,
        d_state: ds,
        d_conv: dc,
        dt_rank: dr,
        xdbl_dim: xd,
        seq_len,
        mamba_input_dim: dm,
        n_layers: nl,
    };

    let n = batch * seq_len * dm;
    let inp = det(n, 0xA1);
    let dt = det(n, 0xB2);

    // ---- EAGER ----
    let mut e_w = GpuMambaTrainWeights::from_cpu(&ctx.stream, &cpu).unwrap();
    let mut e_acts = GpuMambaBackboneActs::new(&ctx.stream, &dims).unwrap();
    let mut e_scratch = GpuMambaScratch::new(&ctx.stream, &dims).unwrap();
    let mut e_state = GpuRecurrentState {
        conv_states: GpuBuffer::zeros(&ctx.stream, nl * di * dc).unwrap(),
        ssm_states: GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap(),
        a_neg_all: GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap(),
    };
    let mut a_neg_flat = vec![0.0f32; nl * di * ds];
    for (l, lw) in cpu.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg_flat[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }
    let mut e_a_neg = GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap();
    e_a_neg.upload(&ctx.stream, &a_neg_flat).unwrap();
    e_state.a_neg_all.upload(&ctx.stream, &a_neg_flat).unwrap();
    let mut e_temp = GpuBuffer::zeros(&ctx.stream, n).unwrap();
    let mut e_input = GpuBuffer::zeros(&ctx.stream, n).unwrap();
    let mut e_dtemp = GpuBuffer::zeros(&ctx.stream, n).unwrap();
    let mut e_grads = GpuMambaGrads::new(&ctx.stream, &cfg, dm).unwrap();
    let mut e_adam = GpuAdamW::new(&ctx.stream, e_grads.flat.len())
        .unwrap()
        .with_lr(1e-4)
        .with_weight_decay(1e-2);
    let mut e_bias = AdamWBiasFactors::new(&ctx.stream).unwrap();

    e_input.upload(&ctx.stream, &inp).unwrap();
    e_dtemp.upload(&ctx.stream, &dt).unwrap();
    e_grads.zero(&ctx.stream).unwrap();
    gpu_forward_mamba_backbone(
        &ctx,
        &mut e_temp,
        &mut e_acts,
        &e_w,
        &e_input,
        &mut e_state,
        &mut e_scratch,
    )
    .unwrap();
    gpu_backward_mamba_backbone(
        &ctx,
        &mut e_dtemp,
        &e_grads,
        &e_acts,
        &e_w,
        &e_a_neg,
        &mut e_scratch,
    )
    .unwrap();
    let (_, bc1, bc2) = e_adam.advance();
    e_bias.write(&ctx.stream, bc1, bc2).unwrap();
    step_m1_capturable(
        &ctx,
        &ctx.kernels.adamw_step_f32_capturable,
        &e_adam,
        e_bias.ptr(),
        &mut e_w,
        &e_grads,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let after_eager = e_w.norm_f_weight.to_cpu(&ctx.stream).unwrap();
    let layer0_in_proj_eager = e_w.layers[0].in_proj_w.to_cpu(&ctx.stream).unwrap();

    // ---- GRAPH ----
    let mut g_w = GpuMambaTrainWeights::from_cpu(&ctx.stream, &cpu).unwrap();
    let mut g_acts = GpuMambaBackboneActs::new(&ctx.stream, &dims).unwrap();
    let mut g_scratch = GpuMambaScratch::new(&ctx.stream, &dims).unwrap();
    let mut g_state = GpuRecurrentState {
        conv_states: GpuBuffer::zeros(&ctx.stream, nl * di * dc).unwrap(),
        ssm_states: GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap(),
        a_neg_all: GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap(),
    };
    g_state.a_neg_all.upload(&ctx.stream, &a_neg_flat).unwrap();
    let mut g_a_neg = GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap();
    g_a_neg.upload(&ctx.stream, &a_neg_flat).unwrap();
    let mut g_temp = GpuBuffer::zeros(&ctx.stream, n).unwrap();
    let mut g_input = GpuBuffer::zeros(&ctx.stream, n).unwrap();
    let mut g_dtemp = GpuBuffer::zeros(&ctx.stream, n).unwrap();
    let mut g_grads = GpuMambaGrads::new(&ctx.stream, &cfg, dm).unwrap();
    let mut g_adam = GpuAdamW::new(&ctx.stream, g_grads.flat.len())
        .unwrap()
        .with_lr(1e-4)
        .with_weight_decay(1e-2);
    let mut g_bias = AdamWBiasFactors::new(&ctx.stream).unwrap();
    g_input.upload(&ctx.stream, &inp).unwrap();
    g_dtemp.upload(&ctx.stream, &dt).unwrap();
    g_bias.write(&ctx.stream, 1.0, 1.0).unwrap();

    let graph = GpuMambaF32TrainingStepGraph::capture(
        &ctx,
        &mut g_w,
        &g_adam,
        &g_bias,
        &mut g_grads,
        &mut g_acts,
        &mut g_scratch,
        &g_a_neg,
        &mut g_temp,
        &g_input,
        &mut g_dtemp,
        &mut g_state,
        batch,
        seq_len,
    )
    .unwrap();
    let (_, bc1, bc2) = g_adam.advance();
    g_bias.write(&ctx.stream, bc1, bc2).unwrap();
    graph
        .replay(
            &g_w, &g_adam, &g_bias, &g_grads, &g_temp, &g_a_neg, &g_input, &g_dtemp, &g_state,
        )
        .unwrap();
    ctx.stream.synchronize().unwrap();
    let after_graph = g_w.norm_f_weight.to_cpu(&ctx.stream).unwrap();
    let layer0_in_proj_graph = g_w.layers[0].in_proj_w.to_cpu(&ctx.stream).unwrap();

    let max_norm_f = after_eager
        .iter()
        .zip(&after_graph)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_in_proj = layer0_in_proj_eager
        .iter()
        .zip(&layer0_in_proj_graph)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("M1 f32 graph: max_err norm_f={max_norm_f:.3e} in_proj={max_in_proj:.3e}");
    assert!(max_norm_f < 1e-5);
    assert!(max_in_proj < 1e-5);
}

#[test]
fn m3_f32_training_graph_matches_eager() {
    use mamba_rs::mamba_ssm::gpu::adamw::step_m3_capturable;
    use mamba_rs::mamba3_siso::config::Mamba3Config;
    use mamba_rs::mamba3_siso::gpu::backward::gpu_backward_mamba3_backbone;
    use mamba_rs::mamba3_siso::gpu::forward::gpu_forward_mamba3_backbone;
    use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
    use mamba_rs::mamba3_siso::gpu::state::{
        GpuMamba3BackboneActs, GpuMamba3Dims, GpuMamba3Scratch,
    };
    use mamba_rs::mamba3_siso::gpu::training_graph::GpuMamba3F32TrainingStepGraph;
    use mamba_rs::mamba3_siso::gpu::weights::{GpuMamba3Grads, GpuMamba3Weights};
    use mamba_rs::mamba3_siso::weights::Mamba3Weights;

    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    };
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = Mamba3Kernels::compile(dev.context(), "sm_89").unwrap();
    let batch = 1;
    let seq_len = 64;
    let dm = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let nh = cfg.nheads();
    let hd = cfg.headdim;
    let na = cfg.num_rope_angles().max(1);
    let nl = cfg.n_layers;

    // f32 M3 forward also has no identity-proj branch. Keep input_proj_w
    // populated as eye(dm) so f32 forward's GEMM has valid operands.
    let mut cpu = Mamba3Weights::init(&cfg, dm, 0xF3F0017A);
    cpu.input_proj_w = (0..dm * dm)
        .map(|i| if i / dm == i % dm { 1.0 } else { 0.0 })
        .collect();
    cpu.input_proj_b = vec![0.0; dm];

    let dims = GpuMamba3Dims {
        batch,
        d_model: dm,
        d_inner: di,
        d_state: ds,
        nheads: nh,
        headdim: hd,
        ngroups: cfg.ngroups,
        in_proj_dim: cfg.in_proj_out_dim(),
        seq_len,
        mamba_input_dim: dm,
        n_layers: nl,
        n_angles: cfg.num_rope_angles(),
        a_floor: cfg.a_floor,
        is_outproj_norm: cfg.is_outproj_norm,
        use_parallel_scan: true,
    };

    let n = batch * seq_len * dm;
    let inp = det(n, 0xC1);
    let dt = det(n, 0xD2);

    // Helper to build a fresh training environment — reused for eager + graph.
    let make = || {
        let w = GpuMamba3Weights::from_cpu(&ctx.stream, &cpu, &cfg, dm).unwrap();
        let acts = GpuMamba3BackboneActs::new(&ctx.stream, &dims).unwrap();
        let scratch = GpuMamba3Scratch::new(&ctx.stream, &dims).unwrap();
        let temp = GpuBuffer::zeros(&ctx.stream, n).unwrap();
        let mi = GpuBuffer::zeros(&ctx.stream, n).unwrap();
        let dtemp = GpuBuffer::zeros(&ctx.stream, n).unwrap();
        let ssm = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd * ds).unwrap();
        let ks = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * ds).unwrap();
        let vs = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd).unwrap();
        let ang = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * na).unwrap();
        let grads = GpuMamba3Grads::new(&ctx.stream, &cfg, dm).unwrap();
        let adam = GpuAdamW::new(&ctx.stream, grads.flat.len())
            .unwrap()
            .with_lr(1e-4)
            .with_weight_decay(1e-2);
        let bias = AdamWBiasFactors::new(&ctx.stream).unwrap();
        ctx.stream.synchronize().unwrap();
        (
            w, acts, scratch, temp, mi, dtemp, ssm, ks, vs, ang, grads, adam, bias,
        )
    };

    // ---- EAGER ----
    let (
        mut e_w,
        mut e_acts,
        mut e_scratch,
        mut e_temp,
        mut e_mi,
        mut e_dtemp,
        mut e_ssm,
        mut e_ks,
        mut e_vs,
        mut e_ang,
        mut e_grads,
        mut e_adam,
        mut e_bias,
    ) = make();
    e_mi.upload(&ctx.stream, &inp).unwrap();
    e_dtemp.upload(&ctx.stream, &dt).unwrap();
    e_grads.zero(&ctx.stream).unwrap();
    gpu_forward_mamba3_backbone(
        &ctx,
        &m3k,
        &mut e_temp,
        &mut e_acts,
        &e_w,
        &e_mi,
        &mut e_ssm,
        &mut e_ks,
        &mut e_vs,
        &mut e_ang,
        &mut e_scratch,
        &dims,
    )
    .unwrap();
    gpu_backward_mamba3_backbone(
        &ctx,
        &m3k,
        &mut e_dtemp,
        &e_acts,
        &e_w,
        &e_grads,
        &mut e_scratch,
        &dims,
    )
    .unwrap();
    let (_, bc1, bc2) = e_adam.advance();
    e_bias.write(&ctx.stream, bc1, bc2).unwrap();
    step_m3_capturable(
        &ctx,
        &m3k.adamw_step_f32_capturable,
        &e_adam,
        e_bias.ptr(),
        &mut e_w,
        &e_grads,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let after_eager = e_w.norm_f_weight.to_cpu(&ctx.stream).unwrap();
    let in_proj_eager = e_w.layers[0].in_proj_w.to_cpu(&ctx.stream).unwrap();

    // ---- GRAPH ----
    let (
        mut g_w,
        mut g_acts,
        mut g_scratch,
        mut g_temp,
        mut g_mi,
        mut g_dtemp,
        mut g_ssm,
        mut g_ks,
        mut g_vs,
        mut g_ang,
        mut g_grads,
        mut g_adam,
        mut g_bias,
    ) = make();
    g_mi.upload(&ctx.stream, &inp).unwrap();
    g_dtemp.upload(&ctx.stream, &dt).unwrap();
    g_bias.write(&ctx.stream, 1.0, 1.0).unwrap();
    let graph = GpuMamba3F32TrainingStepGraph::capture(
        &ctx,
        &m3k,
        &mut g_w,
        &g_adam,
        &g_bias,
        &mut g_grads,
        &mut g_acts,
        &mut g_scratch,
        &mut g_temp,
        &g_mi,
        &mut g_dtemp,
        &mut g_ssm,
        &mut g_ks,
        &mut g_vs,
        &mut g_ang,
        &dims,
    )
    .unwrap();
    let (_, bc1, bc2) = g_adam.advance();
    g_bias.write(&ctx.stream, bc1, bc2).unwrap();
    graph
        .replay(
            &g_w, &g_adam, &g_bias, &g_grads, &g_temp, &g_mi, &g_dtemp, &g_ssm, &g_ks, &g_vs,
            &g_ang,
        )
        .unwrap();
    ctx.stream.synchronize().unwrap();
    let after_graph = g_w.norm_f_weight.to_cpu(&ctx.stream).unwrap();
    let in_proj_graph = g_w.layers[0].in_proj_w.to_cpu(&ctx.stream).unwrap();

    let max_norm_f = after_eager
        .iter()
        .zip(&after_graph)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_in_proj = in_proj_eager
        .iter()
        .zip(&in_proj_graph)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("M3 f32 graph: max_err norm_f={max_norm_f:.3e} in_proj={max_in_proj:.3e}");
    assert!(max_norm_f < 1e-5);
    assert!(max_in_proj < 1e-5);
}
