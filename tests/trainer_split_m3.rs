//! M3 split-step API contracts: `Mamba3Trainer::forward` / `backward_step`
//! against the fused `step()` — the M3 twins of `trainer_split.rs`.
//!
//! Same specification: the split is an EAGER re-composition of the exact
//! eager phase bodies the fused step runs, so split == fused must be
//! bit-identical on the master weights; the accumulate_only window must be
//! exact and the fused step must refuse while it is open; f16 rides the
//! GradScaler protocol and rejects accumulate_only.

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::trainer::{BackwardOpts, Mamba3Trainer, TrainSessionCfg};
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

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

fn test_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
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

/// Bit-exact comparison of two M3 weight snapshots, tensor by tensor.
fn assert_snapshots_bit_eq(a: &Mamba3Weights, b: &Mamba3Weights, ctx_msg: &str) {
    let mut pairs: Vec<(&str, &[f32], &[f32])> = vec![
        ("input_proj_w", &a.input_proj_w, &b.input_proj_w),
        ("input_proj_b", &a.input_proj_b, &b.input_proj_b),
        ("norm_f_weight", &a.norm_f_weight, &b.norm_f_weight),
    ];
    for (la, lb) in a.layers.iter().zip(b.layers.iter()) {
        pairs.extend([
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
            ("dt_bias", la.dt_bias.as_slice(), lb.dt_bias.as_slice()),
            (
                "b_norm_weight",
                la.b_norm_weight.as_slice(),
                lb.b_norm_weight.as_slice(),
            ),
            (
                "c_norm_weight",
                la.c_norm_weight.as_slice(),
                lb.c_norm_weight.as_slice(),
            ),
            ("b_bias", la.b_bias.as_slice(), lb.b_bias.as_slice()),
            ("c_bias", la.c_bias.as_slice(), lb.c_bias.as_slice()),
            ("d_param", la.d_param.as_slice(), lb.d_param.as_slice()),
            (
                "norm_gate_weight",
                la.norm_gate_weight.as_slice(),
                lb.norm_gate_weight.as_slice(),
            ),
            (
                "out_proj_w",
                la.out_proj_w.as_slice(),
                lb.out_proj_w.as_slice(),
            ),
        ]);
    }
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

    let mut w = Mamba3Weights::init(&cfg, input_dim, 0xC0FFEE);
    if !matches!(dtype, WeightDtype::F32) {
        // The M3 mixed pipeline supports the identity D2D branch only.
        w.input_proj_w.clear();
        w.input_proj_b.clear();
    }

    let mut fused = Mamba3Trainer::new_full(
        0,
        &w,
        cfg.clone(),
        session(batch, seq_len, input_dim),
        dtype,
    )
    .expect("fused trainer");
    let mut split = Mamba3Trainer::new_full(0, &w, cfg, session(batch, seq_len, input_dim), dtype)
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
    assert_snapshots_bit_eq(&sa, &sb, &format!("M3 split-vs-fused {dtype:?}"));
}

#[test]
fn m3_split_matches_fused_bit_identical_f32() {
    run_split_vs_fused(WeightDtype::F32);
}

#[test]
fn m3_split_matches_fused_bit_identical_bf16() {
    run_split_vs_fused(WeightDtype::Bf16);
}

/// Interlock + accumulation window semantics (F32).
#[test]
fn m3_split_interlock_and_accumulate_window() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let w = Mamba3Weights::init(&cfg, input_dim, 0xC0FFEE);
    let mut t = Mamba3Trainer::new_full(
        0,
        &w,
        cfg.clone(),
        session(batch, seq_len, input_dim),
        WeightDtype::F32,
    )
    .expect("trainer");

    let n_in = batch * seq_len * input_dim;
    let n_out = batch * seq_len * cfg.d_model;
    let input = det(n_in, 0xAA, 0.05);
    let d_temporal = det(n_out, 0xBB, 0.01);
    let mut out = vec![0.0f32; n_out];

    // Backward with no pending forward -> Err.
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

    // Close the window with a clip; the fused step works again.
    t.forward(&input, &mut out).expect("forward 2");
    let m2 = t
        .backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(1.0))
        .expect("closing backward");
    assert!(m2.optimizer_stepped);
    assert_eq!(m2.step, 1);
    let norm = m2.grad_norm.expect("clip requested -> norm reported");
    assert!(norm.is_finite() && norm > 0.0, "grad norm: {norm}");
    t.step(&input, &d_temporal).expect("fused step after close");
}

/// f16 split: GradScaler protocol rides backward_step; accumulate_only errs.
#[test]
fn m3_f16_split_scaler_protocol() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let mut w = Mamba3Weights::init(&cfg, input_dim, 0xC0FFEE);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    let mut t = Mamba3Trainer::new_full(
        0,
        &w,
        cfg.clone(),
        session(batch, seq_len, input_dim),
        WeightDtype::F16,
    )
    .expect("f16 trainer");

    let n_in = batch * seq_len * input_dim;
    let n_out = batch * seq_len * cfg.d_model;
    let input = det(n_in, 0xAC, 0.05);
    let d_temporal = det(n_out, 0xBC, 0.01);
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
    // The rejected call must not have consumed the pending forward.
    let m = t
        .backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(1.0))
        .expect("f16 backward");
    assert!(m.loss_scale.is_some());
    assert!(m.overflow_skipped.is_some());
    if m.optimizer_stepped {
        let norm = m.grad_norm.expect("clean step reports the norm");
        assert!(norm.is_finite() && norm > 0.0);
    }
}
