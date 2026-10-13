//! M3 split-step API contracts: `Mamba3GpuTrainer::forward` / `backward_step`
//! against the fused `step()` — the M3 twins of `trainer_split.rs`.
//!
//! Same specification: the split is an EAGER re-composition of the exact
//! eager phase bodies the fused step runs, so split == fused must be
//! bit-identical on the master weights; the accumulate_only window must be
//! exact and the fused step must refuse while it is open; f16 rides the
//! GradScaler protocol and rejects accumulate_only.

#![cfg(feature = "cuda")]

use mamba_rs::gpu::GemmMode;
use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::trainer::{BackwardOpts, Mamba3GpuTrainer, TrainSessionCfg};
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
        ..Mamba3Config::default()
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

    let mut fused = Mamba3GpuTrainer::new(
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        GpuOptions::default().dtype(dtype),
    )
    .expect("fused trainer");
    let mut split = Mamba3GpuTrainer::new(
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        GpuOptions::default().dtype(dtype),
    )
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
    let mut t = Mamba3GpuTrainer::new(
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        GpuOptions::default().dtype(WeightDtype::F32),
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

fn assert_split_route_change_rejected(dtype: WeightDtype) {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let mut w = Mamba3Weights::init(&cfg, input_dim, 0xA11C_E002);
    if !matches!(dtype, WeightDtype::F32) {
        w.input_proj_w.clear();
        w.input_proj_b.clear();
    }
    let mut trainer = Mamba3GpuTrainer::new(
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        GpuOptions::default().dtype(dtype),
    )
    .expect("trainer");
    let input = det(batch * seq_len * input_dim, 0xA2, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xB2, 0.01);
    let mut output = vec![0.0; batch * seq_len * cfg.d_model];

    trainer.forward(&input, &mut output).expect("split forward");
    trainer.ctx().set_gemm_mode(GemmMode::CublasFast).unwrap();
    let error = trainer
        .backward_step(&d_temporal, BackwardOpts::default())
        .expect_err("route drift must reject saved activations");
    assert!(
        error.starts_with("M3 split backward: GEMM route changed since forward"),
        "unexpected route error: {error}"
    );
    let second = trainer
        .backward_step(&d_temporal, BackwardOpts::default())
        .expect_err("a rejected backward must close the split lifecycle");
    assert!(second.starts_with("backward_step() without a pending forward()"));
}

#[test]
fn m3_split_rejects_gemm_route_change_bf16() {
    assert_split_route_change_rejected(WeightDtype::Bf16);
}

#[test]
fn m3_split_rejects_gemm_route_change_f32() {
    assert_split_route_change_rejected(WeightDtype::F32);
}

fn assert_reset_closes_split_lifecycle(dtype: WeightDtype) {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let mut w = Mamba3Weights::init(&cfg, input_dim, 0xA11C_E003);
    if !matches!(dtype, WeightDtype::F32) {
        w.input_proj_w.clear();
        w.input_proj_b.clear();
    }
    let mut trainer = Mamba3GpuTrainer::new(
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        GpuOptions::default().dtype(dtype),
    )
    .expect("trainer");
    let input = det(batch * seq_len * input_dim, 0xA3, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xB3, 0.01);
    let mut output = vec![0.0; batch * seq_len * cfg.d_model];

    trainer.forward(&input, &mut output).expect("split forward");
    trainer.reset_state().expect("reset state");
    let error = trainer
        .backward_step(&d_temporal, BackwardOpts::default())
        .expect_err("reset_state must discard saved split activations");
    assert!(
        error.starts_with("backward_step() without a pending forward()"),
        "unexpected lifecycle error: {error}"
    );
}

#[test]
fn m3_reset_closes_split_lifecycle_bf16() {
    assert_reset_closes_split_lifecycle(WeightDtype::Bf16);
}

#[test]
fn m3_reset_closes_split_lifecycle_f32() {
    assert_reset_closes_split_lifecycle(WeightDtype::F32);
}

fn assert_failed_fused_step_closes_split_lifecycle(dtype: WeightDtype) {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let mut w = Mamba3Weights::init(&cfg, input_dim, 0xA11C_E004);
    if !matches!(dtype, WeightDtype::F32) {
        w.input_proj_w.clear();
        w.input_proj_b.clear();
    }
    let mut trainer = Mamba3GpuTrainer::new(
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        GpuOptions::default().dtype(dtype),
    )
    .expect("trainer");
    let input = det(batch * seq_len * input_dim, 0xA4, 0.05);
    let d_temporal = det(batch * seq_len * cfg.d_model, 0xB4, 0.01);
    let mut output = vec![0.0; batch * seq_len * cfg.d_model];

    trainer
        .forward(&input, &mut output)
        .expect("split forward 1");
    trainer
        .backward_step(
            &d_temporal,
            BackwardOpts::default().with_accumulate_only(true),
        )
        .expect("open accumulation window");
    trainer
        .forward(&input, &mut output)
        .expect("split forward 2");
    trainer
        .step(&input, &d_temporal)
        .expect_err("fused step must reject an open accumulation window");
    let error = trainer
        .backward_step(&d_temporal, BackwardOpts::default())
        .expect_err("a failed fused step must discard saved split activations");
    assert!(
        error.starts_with("backward_step() without a pending forward()"),
        "unexpected lifecycle error: {error}"
    );
}

#[test]
fn m3_failed_fused_step_closes_split_lifecycle_bf16() {
    assert_failed_fused_step_closes_split_lifecycle(WeightDtype::Bf16);
}

#[test]
fn m3_failed_fused_step_closes_split_lifecycle_f32() {
    assert_failed_fused_step_closes_split_lifecycle(WeightDtype::F32);
}

/// f16 split: the GradScaler protocol rides backward_step, and an
/// accumulation window keeps one loss scale. The accumulating call steps
/// nothing and reports the scale; the applying call steps once over the
/// window's sum at that same scale and reports the clip norm.
#[test]
fn m3_f16_split_scaler_protocol() {
    let (mut t, input, d_temporal, mut out) = f16_trainer();
    t.forward(&input, &mut out).expect("forward");
    let a = t
        .backward_step(
            &d_temporal,
            BackwardOpts::default().with_accumulate_only(true),
        )
        .expect("f16 accumulating backward");
    assert!(!a.optimizer_stepped);
    assert_eq!(a.overflow_skipped, None);
    let scale = a.loss_scale.expect("the window's loss scale");
    t.forward(&input, &mut out).expect("forward 2");
    let m = t
        .backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(1.0))
        .expect("f16 applying backward");
    assert_eq!(
        m.loss_scale,
        Some(scale),
        "the loss scale holds for the window"
    );
    assert_eq!(m.overflow_skipped, Some(false));
    assert!(m.optimizer_stepped);
    assert_eq!(m.step, 1);
    let norm = m.grad_norm.expect("a clean step reports the norm");
    assert!(norm.is_finite() && norm > 0.0);
}

/// An overflow in one micro-batch survives the window's sum: the applying
/// call skips the whole window, leaves every weight and the step where
/// they were, and backs the loss scale off for the next window, which
/// steps. `apply_step` closes a window the same way.
#[test]
fn m3_f16_overflow_in_a_micro_batch_skips_the_window() {
    let (mut t, input, d_temporal, mut out) = f16_trainer();
    let before = t.snapshot_master().expect("snapshot before");
    let mut blown = d_temporal.clone();
    blown[0] = f32::MAX;
    t.forward(&input, &mut out).expect("forward");
    let a = t
        .backward_step(&blown, BackwardOpts::default().with_accumulate_only(true))
        .expect("f16 accumulating backward");
    let scale = a.loss_scale.expect("the window's loss scale");
    t.forward(&input, &mut out).expect("forward 2");
    let m = t
        .backward_step(&d_temporal, BackwardOpts::default())
        .expect("f16 applying backward");
    assert_eq!(m.overflow_skipped, Some(true));
    assert!(!m.optimizer_stepped);
    assert_eq!(m.step, 0);
    assert_snapshots_bit_eq(
        &before,
        &t.snapshot_master().expect("snapshot after"),
        "a skipped window",
    );
    t.forward(&input, &mut out).expect("forward 3");
    t.backward_step(
        &d_temporal,
        BackwardOpts::default().with_accumulate_only(true),
    )
    .expect("f16 accumulating backward 2");
    let n = t.apply_step(None).expect("f16 apply_step");
    let backed_off = n.loss_scale.expect("the next window's scale");
    assert!(
        backed_off < scale,
        "{backed_off} after an overflow at {scale}"
    );
    assert_eq!(n.overflow_skipped, Some(false));
    assert!(n.optimizer_stepped);
    assert_eq!(n.step, 1);
}

/// An f16 trainer on the identity input projection, with an input and an
/// output gradient for one window of one sequence.
fn f16_trainer() -> (Mamba3GpuTrainer, Vec<f32>, Vec<f32>, Vec<f32>) {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let mut w = Mamba3Weights::init(&cfg, input_dim, 0xC0FFEE);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    let t = Mamba3GpuTrainer::new(
        &w,
        cfg,
        session(batch, seq_len, input_dim),
        GpuOptions::default().dtype(WeightDtype::F16),
    )
    .expect("f16 trainer");
    let n_in = batch * seq_len * input_dim;
    let n_out = batch * seq_len * cfg.d_model;
    (
        t,
        det(n_in, 0xAC, 0.05),
        det(n_out, 0xBC, 0.01),
        vec![0.0f32; n_out],
    )
}

/// Two accumulated micro-batches (batch 1, state reset between) against one
/// big batch (batch 2) of the same samples: the gradients sum either way,
/// so the weights agree within the tolerance of a different GEMM batch
/// shape. In f16 both runs hold the same loss scale, so the window sums
/// the same scaled gradients the big batch takes.
fn assert_accumulation_two_micro_equals_one_big_batch(dtype: WeightDtype, tol: f32) {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let seq_len = 8usize;
    let w = Mamba3Weights::init(&cfg, input_dim, 0xC0FFEE);
    let n_in = seq_len * input_dim;
    let n_out = seq_len * cfg.d_model;
    let (s1_in, s2_in) = (det(n_in, 0xA1, 0.05), det(n_in, 0xA2, 0.05));
    let (s1_dt, s2_dt) = (det(n_out, 0xB1, 0.01), det(n_out, 0xB2, 0.01));

    let mut big = Mamba3GpuTrainer::new(
        &w,
        cfg,
        session(2, seq_len, input_dim),
        GpuOptions::default().dtype(dtype),
    )
    .expect("big trainer");
    big.ctx().set_gemm_mode(GemmMode::Deterministic).unwrap();
    big.load_scaler_state(F16_TEST_SCALE, 0);
    let big_in: Vec<f32> = s1_in.iter().chain(&s2_in).copied().collect();
    let big_dt: Vec<f32> = s1_dt.iter().chain(&s2_dt).copied().collect();
    let mut big_out = vec![0.0f32; 2 * n_out];
    big.forward(&big_in, &mut big_out).expect("big forward");
    let stepped = big
        .backward_step(&big_dt, BackwardOpts::default())
        .expect("big backward")
        .optimizer_stepped;
    assert!(stepped, "{dtype:?}: the big batch stepped");
    let sa = big.snapshot_master().expect("big snapshot");

    let mut micro = Mamba3GpuTrainer::new(
        &w,
        cfg,
        session(1, seq_len, input_dim),
        GpuOptions::default().dtype(dtype),
    )
    .expect("micro trainer");
    micro.ctx().set_gemm_mode(GemmMode::Deterministic).unwrap();
    micro.load_scaler_state(F16_TEST_SCALE, 0);
    let mut out = vec![0.0f32; n_out];
    micro.forward(&s1_in, &mut out).expect("micro forward 1");
    micro
        .backward_step(&s1_dt, BackwardOpts::default().with_accumulate_only(true))
        .expect("micro backward 1");
    micro.reset_state().expect("reset");
    micro.forward(&s2_in, &mut out).expect("micro forward 2");
    let stepped = micro
        .backward_step(&s2_dt, BackwardOpts::default())
        .expect("micro backward 2")
        .optimizer_stepped;
    assert!(stepped, "{dtype:?}: the window stepped");
    let sb = micro.snapshot_master().expect("micro snapshot");

    let flat = |s: &Mamba3Weights| -> Vec<f32> {
        let mut v = Vec::new();
        v.extend_from_slice(&s.input_proj_w);
        v.extend_from_slice(&s.input_proj_b);
        v.extend_from_slice(&s.norm_f_weight);
        for l in &s.layers {
            for t in [
                &l.norm_weight,
                &l.in_proj_w,
                &l.dt_bias,
                &l.b_norm_weight,
                &l.c_norm_weight,
                &l.b_bias,
                &l.c_bias,
                &l.d_param,
                &l.norm_gate_weight,
                &l.out_proj_w,
            ] {
                v.extend_from_slice(t);
            }
        }
        v
    };
    let mut worst = 0.0f32;
    for (a, b) in flat(&sa).iter().zip(flat(&sb).iter()) {
        let d = (a - b).abs();
        worst = worst.max(d / a.abs().max(b.abs()).max(1e-4));
    }
    eprintln!("{dtype:?} accumulation vs big batch: max_rel={worst:e}");
    assert!(
        worst < tol,
        "{dtype:?}: accumulated micro-batches diverge from the big batch: max_rel={worst:e}"
    );
}

#[test]
fn m3_accumulation_two_micro_equals_one_big_batch() {
    assert_accumulation_two_micro_equals_one_big_batch(WeightDtype::F32, 1e-5);
}

#[test]
fn m3_f16_accumulation_two_micro_equals_one_big_batch() {
    assert_accumulation_two_micro_equals_one_big_batch(WeightDtype::F16, 1e-4);
}

/// The f16 loss scale the accumulation tests start from. At the scaler's
/// own starting scale of 2^16, the gradients these small activations give
/// through the RMSNorm backward pass f16's range and the step is skipped,
/// which is the scaler doing its job but leaves nothing to compare;
/// 2^10 keeps them in range. No effect in the other precisions.
const F16_TEST_SCALE: f32 = 1024.0;

/// A window of two calls on the same sequence steps on twice its gradient.
/// Doubling is exact, the backward is linear in its upstream gradient, and
/// over one short sample every gradient element takes a single add per
/// call, so in f32, tf32 and bf16 the window gives the bits of one call
/// with the gradient doubled. f16 holds one loss scale over the window, but
/// doubling commutes with its rounding only above its subnormal range
/// (2^-14): a gradient intermediate below it rounds differently in the
/// doubled call. There the first moments, which carry the gradient, agree
/// within f16's precision, where a window that dropped or averaged a call
/// would be off by half.
fn assert_window_of_two_is_one_doubled_call(dtype: WeightDtype) {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let (batch, seq_len) = (1usize, 4usize);
    let w = Mamba3Weights::init(&cfg, input_dim, 0xC0FFEE);
    let n_in = batch * seq_len * input_dim;
    let n_out = batch * seq_len * cfg.d_model;
    let input = det(n_in, 0xA5, 0.05);
    let d = det(n_out, 0xB5, 0.01);
    let d2: Vec<f32> = d.iter().map(|v| 2.0 * v).collect();
    let mut out = vec![0.0f32; n_out];
    let trainer = || {
        let mut t = Mamba3GpuTrainer::new(
            &w,
            cfg,
            session(batch, seq_len, input_dim),
            GpuOptions::default().dtype(dtype),
        )
        .expect("trainer");
        t.ctx().set_gemm_mode(GemmMode::Deterministic).unwrap();
        t.load_scaler_state(F16_TEST_SCALE, 0);
        t
    };
    let mut window = trainer();
    window.forward(&input, &mut out).expect("forward 1");
    let a = window
        .backward_step(&d, BackwardOpts::default().with_accumulate_only(true))
        .expect("accumulating backward");
    window.reset_state().expect("reset");
    window.forward(&input, &mut out).expect("forward 2");
    let m = window
        .backward_step(&d, BackwardOpts::default())
        .expect("applying backward");
    assert!(m.optimizer_stepped, "{dtype:?}: {m:?}");
    assert_eq!(m.loss_scale, a.loss_scale, "{dtype:?}: the window's scale");
    let mut doubled = trainer();
    doubled.forward(&input, &mut out).expect("forward");
    let md = doubled
        .backward_step(&d2, BackwardOpts::default())
        .expect("doubled backward");
    assert!(md.optimizer_stepped, "{dtype:?}: the doubled call: {md:?}");
    let label = format!("{dtype:?}: a window of two against one doubled call");
    if matches!(dtype, WeightDtype::F16) {
        assert_first_moments_close(
            &window.optimizer_state().expect("window state").m,
            &doubled.optimizer_state().expect("doubled state").m,
            &label,
        );
    } else {
        assert_snapshots_bit_eq(
            &window.snapshot_master().expect("window snapshot"),
            &doubled.snapshot_master().expect("doubled snapshot"),
            &label,
        );
    }
}

/// Two runs' first moments agree within f16's precision, 2^-10 of the
/// largest of them.
fn assert_first_moments_close(a: &[f32], b: &[f32], ctx_msg: &str) {
    assert_eq!(a.len(), b.len(), "{ctx_msg}");
    let top = a.iter().fold(0.0f32, |t, v| t.max(v.abs()));
    let worst = a
        .iter()
        .zip(b)
        .fold(0.0f32, |t, (x, y)| t.max((x - y).abs()));
    assert!(
        top > 0.0 && worst <= top / 1024.0,
        "{ctx_msg}: the first moments differ by {worst:e} against {top:e}"
    );
}

#[test]
fn m3_window_of_two_is_one_doubled_call_f32() {
    assert_window_of_two_is_one_doubled_call(WeightDtype::F32);
}

#[test]
fn m3_window_of_two_is_one_doubled_call_tf32() {
    assert_window_of_two_is_one_doubled_call(WeightDtype::Tf32);
}

#[test]
fn m3_window_of_two_is_one_doubled_call_bf16() {
    assert_window_of_two_is_one_doubled_call(WeightDtype::Bf16);
}

#[test]
fn m3_window_of_two_is_one_doubled_call_f16() {
    assert_window_of_two_is_one_doubled_call(WeightDtype::F16);
}
