//! Global-norm gradient clipping contracts (BackwardOpts::clip_max_norm).
//!
//! - the reported norm matches an f64 CPU reference over the same gradients;
//! - clip above threshold is a bit-exact no-op on the weight trajectory;
//! - clip below threshold scales gradients exactly like pre-scaling
//!   `d_temporal` by the clip coefficient (backward is linear in
//!   `d_temporal`);
//! - clip + accumulate_only is rejected;
//! - f16: the norm is computed AFTER the unscale (a wrong order shows up as
//!   a ~loss_scale-times inflated norm);
//! - the norm is deterministic across identical runs.

#![cfg(feature = "cuda")]

use mamba_rs::config::MambaConfig;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::trainer::{BackwardOpts, MambaTrainer, TrainSessionCfg};
use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::weights::MambaWeights;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
        })
        .collect()
}

fn test_cfg() -> MambaConfig {
    MambaConfig {
        d_model: 32,
        n_layers: 2,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    }
}

fn session(batch: usize, seq_len: usize, input_dim: usize) -> TrainSessionCfg {
    TrainSessionCfg {
        input_dim,
        batch,
        seq_len,
        lr: 1e-3,
        weight_decay: 1e-2,
    }
}

fn train_weights_from(w: &MambaWeights) -> TrainMambaWeights {
    TrainMambaWeights {
        input_proj_w: w.input_proj_w.clone(),
        input_proj_b: w.input_proj_b.clone(),
        layers: w
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
        norm_f_weight: w.norm_f_weight.clone(),
    }
}

fn flat_weights(s: &MambaWeights) -> Vec<f32> {
    let mut v = Vec::new();
    v.extend_from_slice(&s.input_proj_w);
    v.extend_from_slice(&s.input_proj_b);
    for l in &s.layers {
        v.extend_from_slice(&l.norm_weight);
        v.extend_from_slice(&l.in_proj_w);
        v.extend_from_slice(&l.conv1d_weight);
        v.extend_from_slice(&l.conv1d_bias);
        v.extend_from_slice(&l.x_proj_w);
        v.extend_from_slice(&l.dt_proj_w);
        v.extend_from_slice(&l.dt_proj_b);
        v.extend_from_slice(&l.a_log);
        v.extend_from_slice(&l.d_param);
        v.extend_from_slice(&l.out_proj_w);
    }
    v.extend_from_slice(&s.norm_f_weight);
    v
}

/// The reported grad norm must match an f64 norm of the CPU-computed
/// gradients for the same weights / input / upstream gradient.
#[test]
fn clip_grad_norm_matches_cpu_reference() {
    let cfg = test_cfg();
    let input_dim = 20usize;
    let (batch, seq_len) = (1usize, 8usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xBB, 0.01);

    let mut t = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("trainer");
    let mut out = vec![0.0f32; batch * seq_len * cfg.d_model];
    t.forward(&input, &mut out).expect("forward");
    let m = t
        .backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(1e9))
        .expect("backward");
    let gpu_norm = m.grad_norm.expect("grad_norm reported") as f64;

    // CPU reference gradients.
    let tw = train_weights_from(&w);
    let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
    let di = cfg.d_inner();
    let (ds, dc, nl) = (cfg.d_state, cfg.d_conv, cfg.n_layers);
    let mut a_neg = vec![0.0f32; nl * di * ds];
    for (l, lw) in w.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }
    let mut acts = MambaBackboneFlat::zeros(dims);
    let mut fwd_scratch = PhaseScratch::zeros(&dims);
    let mut conv = vec![0.0f32; nl * di * dc];
    let mut ssm = vec![0.0f32; nl * di * ds];
    let mut state = MambaRecurrentState {
        conv: &mut conv,
        ssm: &mut ssm,
        a_neg: &a_neg,
    };
    let mut temporal = vec![0.0f32; seq_len * cfg.d_model];
    forward_mamba_backbone_batched(
        &mut temporal,
        &mut acts,
        &tw,
        &input,
        &mut state,
        &mut fwd_scratch,
        &dims,
    );
    let mut grads = TrainMambaWeights::zeros_from_dims(&dims);
    let mut d_t = d_temporal.clone();
    let mut bwd_scratch = BackwardPhaseScratch::zeros(&dims);
    backward_mamba_backbone_batched(
        &mut d_t,
        &mut grads,
        &acts,
        &tw,
        &a_neg,
        &mut bwd_scratch,
        &dims,
    );
    let mut flat: Vec<f32> = Vec::new();
    flat.extend_from_slice(&grads.input_proj_w);
    flat.extend_from_slice(&grads.input_proj_b);
    for l in &grads.layers {
        flat.extend_from_slice(&l.norm_weight);
        flat.extend_from_slice(&l.in_proj_w);
        flat.extend_from_slice(&l.conv1d_weight);
        flat.extend_from_slice(&l.conv1d_bias);
        flat.extend_from_slice(&l.x_proj_w);
        flat.extend_from_slice(&l.dt_proj_w);
        flat.extend_from_slice(&l.dt_proj_b);
        flat.extend_from_slice(&l.a_log);
        flat.extend_from_slice(&l.d_param);
        flat.extend_from_slice(&l.out_proj_w);
    }
    flat.extend_from_slice(&grads.norm_f_weight);
    let cpu_norm = flat
        .iter()
        .map(|&g| (g as f64) * (g as f64))
        .sum::<f64>()
        .sqrt();

    let rel = (gpu_norm - cpu_norm).abs() / cpu_norm.max(1e-12);
    assert!(
        rel < 1e-3,
        "grad_norm mismatch: gpu={gpu_norm} cpu={cpu_norm} rel={rel:e}"
    );
}

/// clip far above the actual norm must not change the trajectory at all.
#[test]
fn clip_above_threshold_is_bit_identity() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (2usize, 8usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xBB, 0.01);
    let mut out = vec![0.0f32; batch * seq_len * cfg.d_model];

    let mut a = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("trainer a");
    a.forward(&input, &mut out).expect("fwd a");
    a.backward_step(&d_temporal, BackwardOpts::default())
        .expect("bwd a");

    let mut b = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("trainer b");
    b.forward(&input, &mut out).expect("fwd b");
    let m = b
        .backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(1e9))
        .expect("bwd b");
    assert!(m.grad_norm.expect("norm").is_finite());

    let sa = flat_weights(&a.snapshot_master().expect("a"));
    let sb = flat_weights(&b.snapshot_master().expect("b"));
    for (i, (x, y)) in sa.iter().zip(sb.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "above-threshold clip changed weight [{i}]: {x} vs {y}"
        );
    }
}

/// Backward is linear in d_temporal, so clipping at c must equal running
/// with d_temporal pre-scaled by the clip coefficient.
#[test]
fn clip_below_threshold_equals_prescaled_d_temporal() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 8usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xBB, 0.01);
    let mut out = vec![0.0f32; batch * seq_len * cfg.d_model];

    // Probe the unclipped norm.
    let mut probe = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("probe");
    probe.forward(&input, &mut out).expect("probe fwd");
    let norm = probe
        .backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(1e9))
        .expect("probe bwd")
        .grad_norm
        .expect("norm") as f64;

    let c = (norm / 2.0) as f32;
    let coef = (c as f64 / (norm + 1e-6)) as f32;

    let mut clipped = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("clipped");
    clipped.forward(&input, &mut out).expect("clipped fwd");
    let m = clipped
        .backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(c))
        .expect("clipped bwd");
    let reported = m.grad_norm.expect("norm") as f64;
    assert!(
        (reported - norm).abs() / norm < 1e-6,
        "reported norm must be PRE-clip: {reported} vs {norm}"
    );

    let scaled_dt: Vec<f32> = d_temporal.iter().map(|&v| v * coef).collect();
    let mut prescaled = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("prescaled");
    prescaled.forward(&input, &mut out).expect("prescaled fwd");
    prescaled
        .backward_step(&scaled_dt, BackwardOpts::default())
        .expect("prescaled bwd");

    let sa = flat_weights(&clipped.snapshot_master().expect("clipped"));
    let sb = flat_weights(&prescaled.snapshot_master().expect("prescaled"));
    let mut worst = 0.0f32;
    for (x, y) in sa.iter().zip(sb.iter()) {
        let d = (x - y).abs();
        let denom = x.abs().max(y.abs()).max(1e-4);
        worst = worst.max(d / denom);
    }
    assert!(
        worst < 1e-5,
        "clip-at-c diverges from prescaled d_temporal: max_rel={worst:e}"
    );
}

#[test]
fn clip_with_accumulate_errs() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let mut t = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("trainer");
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xBB, 0.01);
    let mut out = vec![0.0f32; batch * seq_len * cfg.d_model];
    t.forward(&input, &mut out).expect("forward");
    assert!(
        t.backward_step(
            &d_temporal,
            BackwardOpts::default()
                .with_clip_max_norm(1.0)
                .with_accumulate_only(true),
        )
        .is_err(),
        "clip + accumulate_only must be rejected"
    );
}

/// f16: the norm must be computed on UNSCALED gradients. A wrong order
/// inflates the reported norm by ~loss_scale (65536x at init) — comparing
/// against the bf16 trainer's norm (no loss scaler, same identity-input_proj
/// branch) on identical weights catches that class. (The F32 engine cannot
/// serve as the reference here: it has no identity-input_proj branch, and
/// the Mixed engines accept ONLY identity.)
#[test]
fn f16_clip_norm_is_post_unscale() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let mut w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xBB, 0.01);
    let mut out = vec![0.0f32; batch * seq_len * cfg.d_model];

    let mut f16 = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F16,
    )
    .expect("f16 trainer");
    f16.forward(&input, &mut out).expect("f16 fwd");
    let mf = f16
        .backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(1e9))
        .expect("f16 bwd");
    assert_eq!(mf.overflow_skipped, Some(false));
    let f16_norm = mf.grad_norm.expect("f16 norm") as f64;

    let mut bf16 = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::Bf16,
    )
    .expect("bf16 trainer");
    bf16.forward(&input, &mut out).expect("bf16 fwd");
    let bf16_norm = bf16
        .backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(1e9))
        .expect("bf16 bwd")
        .grad_norm
        .expect("bf16 norm") as f64;

    let ratio = f16_norm / bf16_norm;
    assert!(
        (0.5..2.0).contains(&ratio),
        "f16 norm {f16_norm} vs bf16 norm {bf16_norm} (ratio {ratio}) — an inflated \
         ratio means the norm was computed on loss-scaled gradients"
    );
}

/// The norm is deterministic: identical runs report identical bits.
#[test]
fn clip_norm_is_deterministic() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (2usize, 8usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xBB, 0.01);
    let mut out = vec![0.0f32; batch * seq_len * cfg.d_model];

    let mut norms = Vec::new();
    for _ in 0..2 {
        let mut t = MambaTrainer::new_full(
            0,
            &w,
            cfg,
            session(batch, seq_len, input_dim),
            WeightDtype::F32,
        )
        .expect("trainer");
        t.forward(&input, &mut out).expect("fwd");
        let m = t
            .backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(1e9))
            .expect("bwd");
        norms.push(m.grad_norm.expect("norm"));
    }
    assert_eq!(
        norms[0].to_bits(),
        norms[1].to_bits(),
        "grad norm must be bit-deterministic: {} vs {}",
        norms[0],
        norms[1]
    );
}
