//! Split-step API contracts: `MambaTrainer::forward` / `backward_step`
//! against the fused `step()`.
//!
//! The split is specified as an EAGER re-composition of the exact eager
//! phase bodies the fused step runs, so:
//!   - split == fused must be bit-identical on the master weights (F32 and
//!     bf16) — same kernels, same order, only host-side orchestration moves;
//!   - forward() alone must leave the optimizer and weights untouched;
//!   - the accumulate_only window must be exact ("don't zero, don't run the
//!     tail") and the fused step() must refuse to run while it is open
//!     (its captured/eager body zeroes the arena — the zero-state bug class);
//!   - f16 rides the GradScaler protocol and rejects accumulate_only.

#![cfg(feature = "cuda")]

use mamba_rs::config::MambaConfig;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::trainer::{BackwardOpts, MambaTrainer, TrainSessionCfg};
use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::PhaseScratch;
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

/// Bit-exact comparison of two weight snapshots, tensor by tensor.
fn assert_snapshots_bit_eq(a: &MambaWeights, b: &MambaWeights, ctx_msg: &str) {
    let pairs: Vec<(&str, &[f32], &[f32])> = {
        let mut v: Vec<(&str, &[f32], &[f32])> = vec![
            ("input_proj_w", &a.input_proj_w, &b.input_proj_w),
            ("input_proj_b", &a.input_proj_b, &b.input_proj_b),
            ("norm_f_weight", &a.norm_f_weight, &b.norm_f_weight),
        ];
        for (la, lb) in a.layers.iter().zip(b.layers.iter()) {
            v.extend([
                (
                    "norm_weight",
                    la.norm_weight.as_slice(),
                    lb.norm_weight.as_slice(),
                ),
                (
                    "in_proj_w",
                    la.in_proj_w.as_slice(),
                    lb.in_proj_w.as_slice(),
                ),
                (
                    "conv1d_weight",
                    la.conv1d_weight.as_slice(),
                    lb.conv1d_weight.as_slice(),
                ),
                (
                    "conv1d_bias",
                    la.conv1d_bias.as_slice(),
                    lb.conv1d_bias.as_slice(),
                ),
                ("x_proj_w", la.x_proj_w.as_slice(), lb.x_proj_w.as_slice()),
                (
                    "dt_proj_w",
                    la.dt_proj_w.as_slice(),
                    lb.dt_proj_w.as_slice(),
                ),
                (
                    "dt_proj_b",
                    la.dt_proj_b.as_slice(),
                    lb.dt_proj_b.as_slice(),
                ),
                ("a_log", la.a_log.as_slice(), lb.a_log.as_slice()),
                ("d_param", la.d_param.as_slice(), lb.d_param.as_slice()),
                (
                    "out_proj_w",
                    la.out_proj_w.as_slice(),
                    lb.out_proj_w.as_slice(),
                ),
            ]);
        }
        v
    };
    for (name, xa, xb) in pairs {
        assert_eq!(xa.len(), xb.len(), "{ctx_msg}: {name} length mismatch");
        for (i, (va, vb)) in xa.iter().zip(xb.iter()).enumerate() {
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "{ctx_msg}: {name}[{i}] differs: {va} vs {vb}"
            );
        }
    }
}

fn run_split_vs_fused(dtype: WeightDtype) {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (2usize, 8usize);
    let n_in = batch * seq_len * input_dim;
    let n_out = batch * seq_len * cfg.d_model;

    let mut w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    if !matches!(dtype, WeightDtype::F32) {
        // Keep the Mixed variant on the identity D2D branch — this test IS
        // the identity-path regression; the trainable input_proj has its
        // own split-vs-fused test below.
        w.input_proj_w.clear();
        w.input_proj_b.clear();
    }

    let mut fused = MambaTrainer::new_full(0, &w, cfg, session(batch, seq_len, input_dim), dtype)
        .expect("fused trainer");
    let mut split = MambaTrainer::new_full(0, &w, cfg, session(batch, seq_len, input_dim), dtype)
        .expect("split trainer");

    let mut temporal_out = vec![0.0f32; n_out];
    for step in 0..3u32 {
        let input = det(n_in, 0xA0 + step, 0.05);
        let d_temporal = det(n_out, 0xB0 + step, 0.01);

        let fm = fused.step(&input, &d_temporal).expect("fused step");
        assert!(!fm.graph_replayed);

        split
            .forward(&input, &mut temporal_out)
            .expect("split forward");
        let bm = split
            .backward_step(&d_temporal, BackwardOpts::default())
            .expect("split backward");
        assert!(bm.optimizer_stepped);
        assert_eq!(bm.step, fm.step, "adam step counters diverged");
    }

    let sa = fused.snapshot_master().expect("fused snapshot");
    let sb = split.snapshot_master().expect("split snapshot");
    assert_snapshots_bit_eq(&sa, &sb, &format!("split-vs-fused {dtype:?}"));
}

#[test]
fn split_matches_fused_bit_identical_f32() {
    run_split_vs_fused(WeightDtype::F32);
}

#[test]
fn split_matches_fused_bit_identical_bf16() {
    run_split_vs_fused(WeightDtype::Bf16);
}

/// The forward readback must be the post-norm_f training-forward output:
/// compare against the CPU batched training forward on identical weights.
#[test]
fn forward_temporal_matches_cpu_training_forward() {
    let cfg = test_cfg();
    let input_dim = 20usize; // rectangular input_proj — the vision shape
    let (batch, seq_len) = (1usize, 8usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);

    let mut trainer = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("trainer");
    let mut gpu_out = vec![0.0f32; batch * seq_len * cfg.d_model];
    trainer.forward(&input, &mut gpu_out).expect("forward");

    // CPU reference (batch=1: single flat sequence).
    let tw = TrainMambaWeights {
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
    };
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
    let mut scratch = PhaseScratch::zeros(&dims);
    let mut conv = vec![0.0f32; nl * di * dc];
    let mut ssm = vec![0.0f32; nl * di * ds];
    let mut state = MambaRecurrentState {
        conv: &mut conv,
        ssm: &mut ssm,
        a_neg: &a_neg,
    };
    let mut cpu_out = vec![0.0f32; seq_len * cfg.d_model];
    forward_mamba_backbone_batched(
        &mut cpu_out,
        &mut acts,
        &tw,
        &input,
        &mut state,
        &mut scratch,
        &dims,
    );

    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&a, &b) in gpu_out.iter().zip(cpu_out.iter()) {
        dot += a as f64 * b as f64;
        na += (a as f64).powi(2);
        nb += (b as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    assert!(
        cos > 0.9999,
        "GPU forward() vs CPU training forward: cos={cos} (expected > 0.9999)"
    );
}

/// forward() alone must not touch the optimizer or any weight.
#[test]
fn forward_alone_leaves_optimizer_untouched() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (2usize, 8usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let mut trainer = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("trainer");

    let before = trainer.snapshot_master().expect("before");
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let mut out = vec![0.0f32; batch * seq_len * cfg.d_model];
    trainer.forward(&input, &mut out).expect("forward 1");
    trainer.forward(&input, &mut out).expect("forward 2");
    let after = trainer.snapshot_master().expect("after");
    assert_snapshots_bit_eq(&before, &after, "forward-only");
}

/// Interlock: backward without forward errs; the fused step() refuses to run
/// while an accumulate window is open; closing the window restores it.
#[test]
fn split_interlock_and_accumulate_window() {
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

    let n_in = batch * seq_len * input_dim;
    let n_out = batch * seq_len * cfg.d_model;
    let input = det(n_in, 0xAA, 0.05);
    let d_temporal = det(n_out, 0xBB, 0.01);
    let mut out = vec![0.0f32; n_out];

    // Backward with no pending forward → Err.
    assert!(
        t.backward_step(&d_temporal, BackwardOpts::default())
            .is_err()
    );

    // Open an accumulation window.
    t.forward(&input, &mut out).expect("forward");
    let m = t
        .backward_step(
            &d_temporal,
            BackwardOpts::default().with_accumulate_only(true),
        )
        .expect("accumulate backward");
    assert!(!m.optimizer_stepped);
    assert_eq!(m.step, 0, "accumulate-only must not advance adam");

    // Fused step must refuse while the window is open.
    assert!(t.step(&input, &d_temporal).is_err());

    // A second backward needs its own forward.
    assert!(
        t.backward_step(&d_temporal, BackwardOpts::default())
            .is_err()
    );

    // Close the window; the fused step works again.
    t.forward(&input, &mut out).expect("forward 2");
    let m2 = t
        .backward_step(&d_temporal, BackwardOpts::default())
        .expect("closing backward");
    assert!(m2.optimizer_stepped);
    assert_eq!(m2.step, 1);
    t.step(&input, &d_temporal).expect("fused step after close");
}

/// Two accumulated micro-batches (batch=1, state reset between) vs one
/// big batch (batch=2) of the same samples: gradients sum either way, so
/// the resulting weights must agree. Exact bitness across DIFFERENT GEMM
/// batch shapes is reported but asserted only at tight tolerance — the
/// batch-invariant contract is per-bucket row-0 identity, not cross-shape
/// reduction-order identity.
#[test]
fn accumulation_two_micro_equals_one_big_batch() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let seq_len = 8usize;
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);

    let n_in1 = seq_len * input_dim;
    let n_out1 = seq_len * cfg.d_model;
    let s1_in = det(n_in1, 0xA1, 0.05);
    let s2_in = det(n_in1, 0xA2, 0.05);
    let s1_dt = det(n_out1, 0xB1, 0.01);
    let s2_dt = det(n_out1, 0xB2, 0.01);

    // Big batch: both samples in one call.
    let mut big =
        MambaTrainer::new_full(0, &w, cfg, session(2, seq_len, input_dim), WeightDtype::F32)
            .expect("big trainer");
    big.ctx().set_batch_invariant(true);
    let big_in: Vec<f32> = s1_in.iter().chain(s2_in.iter()).copied().collect();
    let big_dt: Vec<f32> = s1_dt.iter().chain(s2_dt.iter()).copied().collect();
    let mut big_out = vec![0.0f32; 2 * n_out1];
    big.forward(&big_in, &mut big_out).expect("big forward");
    big.backward_step(&big_dt, BackwardOpts::default())
        .expect("big backward");
    let sa = big.snapshot_master().expect("big snapshot");

    // Micro: accumulate sample 1, reset state, apply on sample 2.
    let mut micro =
        MambaTrainer::new_full(0, &w, cfg, session(1, seq_len, input_dim), WeightDtype::F32)
            .expect("micro trainer");
    micro.ctx().set_batch_invariant(true);
    let mut out = vec![0.0f32; n_out1];
    micro.forward(&s1_in, &mut out).expect("micro fwd 1");
    micro
        .backward_step(&s1_dt, BackwardOpts::default().with_accumulate_only(true))
        .expect("micro bwd 1");
    micro.reset_state().expect("reset");
    micro.forward(&s2_in, &mut out).expect("micro fwd 2");
    micro
        .backward_step(&s2_dt, BackwardOpts::default())
        .expect("micro bwd 2");
    let sb = micro.snapshot_master().expect("micro snapshot");

    // Tight-tolerance comparison + informational bit report.
    let mut worst = 0.0f32;
    let mut bit_equal = true;
    let flat = |s: &MambaWeights| -> Vec<f32> {
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
    };
    for (a, b) in flat(&sa).iter().zip(flat(&sb).iter()) {
        if a.to_bits() != b.to_bits() {
            bit_equal = false;
        }
        let d = (a - b).abs();
        let denom = a.abs().max(b.abs()).max(1e-4);
        worst = worst.max(d / denom);
    }
    eprintln!("accumulation vs big-batch: bit_equal={bit_equal} max_rel={worst:e}");
    assert!(
        worst < 1e-5,
        "accumulated micro-batches diverge from the big batch: max_rel={worst:e}"
    );
}

fn assert_split_route_change_rejected(dtype: WeightDtype) {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let mut w = MambaWeights::init(&cfg, input_dim, 0xA11C_E001);
    if !matches!(dtype, WeightDtype::F32) {
        w.input_proj_w.clear();
        w.input_proj_b.clear();
    }
    let mut trainer = MambaTrainer::new_full(0, &w, cfg, session(batch, seq_len, input_dim), dtype)
        .expect("trainer");
    let input = det(batch * seq_len * input_dim, 0xA2, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xB2, 0.01);
    let mut output = vec![0.0; batch * seq_len * cfg.d_model];

    trainer.forward(&input, &mut output).expect("split forward");
    trainer.ctx().set_fast_gemm(true);
    let error = trainer
        .backward_step(&d_temporal, BackwardOpts::default())
        .expect_err("route drift must reject saved activations");
    assert!(
        error.starts_with("backward_step(): GEMM route changed since forward()"),
        "unexpected route error: {error}"
    );
    let second = trainer
        .backward_step(&d_temporal, BackwardOpts::default())
        .expect_err("a rejected backward must close the split lifecycle");
    assert!(second.starts_with("backward_step() without a pending forward()"));
}

#[test]
fn split_rejects_gemm_route_change_bf16() {
    assert_split_route_change_rejected(WeightDtype::Bf16);
}

#[test]
fn split_rejects_gemm_route_change_f32() {
    assert_split_route_change_rejected(WeightDtype::F32);
}

/// f16 split: GradScaler protocol rides backward_step; accumulate_only errs.
#[test]
fn f16_split_scaler_protocol() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let mut w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    let mut t = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F16,
    )
    .expect("f16 trainer");

    let n_in = batch * seq_len * input_dim;
    let n_out = batch * seq_len * cfg.d_model;
    let input = det(n_in, 0xAA, 0.05);
    let d_temporal = det(n_out, 0xBB, 0.01);
    let mut out = vec![0.0f32; n_out];

    t.forward(&input, &mut out).expect("forward");
    assert!(
        t.backward_step(
            &d_temporal,
            BackwardOpts::default().with_accumulate_only(true)
        )
        .is_err(),
        "f16 + accumulate_only must be rejected"
    );
    assert!(
        t.backward_step(&d_temporal, BackwardOpts::default())
            .is_err(),
        "a rejected backward must close the pending lifecycle"
    );
    t.forward(&input, &mut out)
        .expect("forward after rejection");
    let m = t
        .backward_step(&d_temporal, BackwardOpts::default())
        .expect("f16 backward");
    assert!(m.loss_scale.is_some());
    assert!(m.overflow_skipped.is_some());
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

/// The first AdamW update is exactly proportional to lr — the
/// `m_hat/(sqrt(v_hat)+eps)` term is lr-independent and the decoupled decay
/// term carries lr too — so set_lr(10x) must scale the first-step weight
/// delta ~10x.
#[test]
fn set_lr_scales_first_step_delta() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xBB, 0.01);
    let w0 = flat_weights(&w);

    let max_delta = |snapshot: &MambaWeights| -> f64 {
        flat_weights(snapshot)
            .iter()
            .zip(w0.iter())
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max)
    };

    let mut a = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("trainer a");
    assert!((a.lr() - 1e-3).abs() < 1e-9);
    a.step(&input, &d_temporal).expect("step a");
    let da = max_delta(&a.snapshot_master().expect("a"));

    let mut b = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("trainer b");
    b.set_lr(1e-2).expect("set_lr");
    assert!((b.lr() - 1e-2).abs() < 1e-9);
    b.step(&input, &d_temporal).expect("step b");
    let db = max_delta(&b.snapshot_master().expect("b"));

    let ratio = db / da;
    assert!(
        (9.5..10.5).contains(&ratio),
        "set_lr(10x) must scale the first-step delta ~10x: got {ratio} ({da} -> {db})"
    );

    assert!(b.set_lr(0.0).is_err(), "lr=0 must be rejected");
    assert!(b.set_lr(f32::NAN).is_err(), "NaN lr must be rejected");
}

/// set_lr under a captured graph APPLIES on the next replay: the lr rides
/// the 3-element device bias buffer ({bc1, bc2, lr}) re-uploaded before
/// every step, so a warmup/cosine schedule works without dropping the
/// graph (the old per-tensor kernels baked lr by value at capture and
/// set_lr had to refuse; that refusal is lifted by the fused AdamW).
/// Proof by effect: two trainers, identical seeds and steps, one raises
/// lr 10x mid-graph — its post-step weight delta scales ~10x.
#[test]
fn set_lr_under_graph_applies_on_next_replay() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xBB, 0.01);

    let run = |lr_after_capture: Option<f32>| -> (Vec<f32>, Vec<f32>) {
        let mut t = MambaTrainer::new_full(
            0,
            &w,
            cfg,
            session(batch, seq_len, input_dim),
            WeightDtype::F32,
        )
        .expect("trainer");
        t.step(&input, &d_temporal).expect("warmup");
        t.capture_graph().expect("capture");
        assert!(t.has_graph());
        if let Some(lr) = lr_after_capture {
            t.set_lr(lr)
                .expect("set_lr under a captured graph must apply");
            assert!((t.lr() - lr).abs() < 1e-12);
        }
        let before = t.snapshot_master().expect("pre");
        let m = t.step(&input, &d_temporal).expect("graph step");
        assert!(m.graph_replayed, "post-capture step must replay the graph");
        let after = t.snapshot_master().expect("post");
        (before.norm_f_weight, after.norm_f_weight)
    };

    // session() configures lr=1e-3 — raise 10x from THAT base, not an
    // assumed constant (a matching literal would make ratio==1 vacuously).
    let base_lr = session(batch, seq_len, input_dim).lr;
    let (b0, a0) = run(None);
    let (b1, a1) = run(Some(base_lr * 10.0));
    let d0 = b0
        .iter()
        .zip(&a0)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let d1 = b1
        .iter()
        .zip(&a1)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(d0 > 0.0, "baseline replay must move weights");
    let ratio = f64::from(d1) / f64::from(d0);
    assert!(
        (5.0..15.0).contains(&ratio),
        "set_lr(10x) under graph must scale the replayed step ~10x: got {ratio} ({d0} -> {d1})"
    );

    // drop_graph still falls back to eager.
    let mut t = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("trainer");
    t.step(&input, &d_temporal).expect("warmup");
    t.capture_graph().expect("capture");
    t.drop_graph();
    assert!(!t.has_graph());
    t.set_lr(5e-4).expect("set_lr after drop_graph");
    let m2 = t.step(&input, &d_temporal).expect("eager step");
    assert!(
        !m2.graph_replayed,
        "after drop_graph steps must run eagerly"
    );
}

/// Reference no-decay groups: with the mask ON, masked tensors (a_log,
/// d_param, dt_proj_b, norm scales) skip the decoupled decay term exactly
/// (w_on - w_off == lr*wd*w0 elementwise), while unmasked tensors stay
/// bit-identical to the mask-OFF run. Default OFF must be the historical
/// behavior.
#[test]
fn reference_no_decay_masks_exactly_the_named_tensors() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xBB, 0.01);
    let (lr, wd) = (1e-3f64, 1e-2f64);

    let mut off = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("off trainer");
    off.step(&input, &d_temporal).expect("off step");
    let s_off = off.snapshot_master().expect("off snapshot");

    let mut on = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("on trainer");
    on.set_reference_no_decay(true).expect("enable mask");
    on.step(&input, &d_temporal).expect("on step");
    let s_on = on.snapshot_master().expect("on snapshot");

    // Unmasked tensors: bit-identical between the two runs.
    let unmasked = |s: &MambaWeights| -> Vec<f32> {
        let mut v = Vec::new();
        v.extend_from_slice(&s.input_proj_w);
        v.extend_from_slice(&s.input_proj_b);
        for l in &s.layers {
            v.extend_from_slice(&l.in_proj_w);
            v.extend_from_slice(&l.conv1d_weight);
            v.extend_from_slice(&l.conv1d_bias);
            v.extend_from_slice(&l.x_proj_w);
            v.extend_from_slice(&l.dt_proj_w);
            v.extend_from_slice(&l.out_proj_w);
        }
        v
    };
    for (i, (a, b)) in unmasked(&s_off)
        .iter()
        .zip(unmasked(&s_on).iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "unmasked tensor element [{i}] changed under the mask: {a} vs {b}"
        );
    }

    // Masked tensors: w_on - w_off == lr*wd*w0 (the decay term), elementwise.
    let masked_with_origin = |s: &MambaWeights, w0: &MambaWeights| -> Vec<(f32, f32)> {
        let mut v: Vec<(f32, f32)> = Vec::new();
        for (l, l0) in s.layers.iter().zip(w0.layers.iter()) {
            v.extend(
                l.norm_weight
                    .iter()
                    .copied()
                    .zip(l0.norm_weight.iter().copied()),
            );
            v.extend(
                l.dt_proj_b
                    .iter()
                    .copied()
                    .zip(l0.dt_proj_b.iter().copied()),
            );
            v.extend(l.a_log.iter().copied().zip(l0.a_log.iter().copied()));
            v.extend(l.d_param.iter().copied().zip(l0.d_param.iter().copied()));
        }
        v.extend(
            s.norm_f_weight
                .iter()
                .copied()
                .zip(w0.norm_f_weight.iter().copied()),
        );
        v
    };
    let off_m = masked_with_origin(&s_off, &w);
    let on_m = masked_with_origin(&s_on, &w);
    let mut any_moved = false;
    for (i, ((a, w0), (b, _))) in off_m.iter().zip(on_m.iter()).enumerate() {
        let expected_gap = lr * wd * (*w0 as f64);
        let gap = (*b as f64) - (*a as f64);
        // Tolerance: the OFF run computes w0*(1 - lr*wd) in f32, whose
        // rounding is ~ulp(w0) = |w0| * 2^-23 — the dominant error term,
        // proportional to |w0| (not to the tiny decay gap itself).
        let tol = 1e-7 + (*w0 as f64).abs() * 5e-7;
        assert!(
            (gap - expected_gap).abs() < tol,
            "masked element [{i}]: gap {gap:e} vs expected decay term {expected_gap:e} \
             (tol {tol:e})"
        );
        if gap.abs() > 0.0 {
            any_moved = true;
        }
    }
    assert!(any_moved, "the mask never changed a single masked element");

    // Toggle under a captured graph must err.
    let mut g = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("graph trainer");
    g.step(&input, &d_temporal).expect("warmup");
    g.capture_graph().expect("capture");
    assert!(g.set_reference_no_decay(true).is_err());
}

/// bf16 with a REAL rectangular input_proj (the vision patch-embed shape):
/// constructs (the guard is gone), trains it (input_proj_w/b move), and the
/// split==fused bit-identity holds through the new branch.
#[test]
fn bf16_trainable_input_proj_split_matches_fused() {
    let cfg = test_cfg();
    let input_dim = 20usize;
    let (batch, seq_len) = (2usize, 8usize);
    let n_in = batch * seq_len * input_dim;
    let n_out = batch * seq_len * cfg.d_model;
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);

    let mut fused = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::Bf16,
    )
    .expect("fused trainer");
    let mut split = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::Bf16,
    )
    .expect("split trainer");
    let mut out = vec![0.0f32; n_out];
    for step in 0..3u32 {
        let input = det(n_in, 0xA0 + step, 0.05);
        let d_temporal = det(n_out, 0xB0 + step, 0.01);
        fused.step(&input, &d_temporal).expect("fused step");
        split.forward(&input, &mut out).expect("split fwd");
        split
            .backward_step(&d_temporal, BackwardOpts::default())
            .expect("split bwd");
    }
    let sa = fused.snapshot_master().expect("fused snapshot");
    let sb = split.snapshot_master().expect("split snapshot");
    assert_snapshots_bit_eq(&sa, &sb, "bf16 rect input_proj split-vs-fused");

    // The patch embed actually trained.
    assert!(
        sa.input_proj_w
            .iter()
            .zip(w.input_proj_w.iter())
            .any(|(a, b)| a != b),
        "input_proj_w never moved over 3 steps"
    );
    assert!(
        sa.input_proj_b
            .iter()
            .zip(w.input_proj_b.iter())
            .any(|(a, b)| a != b),
        "input_proj_b never moved over 3 steps"
    );
}

/// Presize-twin regression (bug #1): a patch dim exceeding every backbone
/// dim under batch-invariant graph capture. Without the input_dim-aware
/// presize, the upcast scratch grows INSIDE capture (illegal alloc) or
/// after it (freed-pointer replay).
#[test]
fn bf16_input_proj_graph_capture_with_large_patch_dim() {
    let cfg = test_cfg(); // d_model=32, d_inner=64 -> 2*d_inner = 128
    let input_dim = 200usize; // exceeds max(d_model, 2*d_inner, xdbl_dim)
    let (batch, seq_len) = (1usize, 4usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let mut t = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::Bf16,
    )
    .expect("trainer");
    t.ctx().set_batch_invariant(true);
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xBB, 0.01);
    t.step(&input, &d_temporal).expect("warmup");
    t.capture_graph().expect("capture");
    let m = t.step(&input, &d_temporal).expect("replay 1");
    assert!(m.graph_replayed);
    t.step(&input, &d_temporal).expect("replay 2");
}

/// Bisection probe: batch-invariant mode + trainable input_proj, EAGER only.
#[test]
fn bf16_input_proj_eager_under_batch_invariant() {
    let cfg = test_cfg();
    let input_dim = 20usize;
    eager_bi_probe(cfg, input_dim);
}

/// Same probe at the large patch dim.
#[test]
fn bf16_input_proj_eager_under_batch_invariant_large() {
    let cfg = test_cfg();
    let input_dim = 200usize;
    eager_bi_probe(cfg, input_dim);
}

/// Forward-only variant of the probe (the split API syncs internally).
#[test]
fn bf16_input_proj_eager_bi_large_forward_only() {
    let cfg = test_cfg();
    let input_dim = 200usize;
    let (batch, seq_len) = (1usize, 4usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let mut t = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::Bf16,
    )
    .expect("trainer");
    t.ctx().set_batch_invariant(true);
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let mut out = vec![0.0f32; batch * seq_len * cfg.d_model];
    t.forward(&input, &mut out).expect("BI forward only");
}

fn eager_bi_probe(cfg: MambaConfig, input_dim: usize) {
    let (batch, seq_len) = (1usize, 4usize);
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    let mut t = MambaTrainer::new_full(
        0,
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        WeightDtype::Bf16,
    )
    .expect("trainer");
    t.ctx().set_batch_invariant(true);
    let input = det(batch * seq_len * input_dim, 0xAA, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xBB, 0.01);
    t.step(&input, &d_temporal).expect("eager BI step");
    t.step(&input, &d_temporal).expect("eager BI step 2");
}
