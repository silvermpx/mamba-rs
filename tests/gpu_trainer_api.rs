//! The GPU trainer surface every family shares.
//!
//! For Mamba-2 and MIMO, `backward_step` is the backward and the AdamW
//! step it replaces, `step` is forward plus `backward_step`, the clip
//! reports the gradient's norm and leaves a gradient under the limit
//! alone, a spiking window is dropped whole, and an accumulation window
//! steps on the sum of its micro-batches. Every family trains through the
//! `GpuTrainer` trait and resumes through its checkpoint and trainer-state
//! files.
#![cfg(feature = "cuda")]

use std::path::Path;

use mamba_rs::config::ScanMode;
use mamba_rs::dist::DistContext;
use mamba_rs::gpu::train_state::TrainerState;
use mamba_rs::{
    AnyGpuTrainer, BackwardOpts, GpuOptions, GpuTrainer, Mamba2Config, Mamba2GpuTrainer,
    Mamba2Weights, Mamba3Config, Mamba3GpuTrainer, Mamba3MimoConfig, Mamba3MimoGpuTrainer,
    Mamba3MimoWeights, Mamba3Weights, MambaConfig, MambaGpuTrainer, MambaWeights, ModelFamily,
    TrainSessionCfg, WeightDtype,
};

const BATCH: usize = 2;
const SEQ_LEN: usize = 48;
const D_MODEL: usize = 32;
const INPUT_DIM: usize = 24;

fn session() -> TrainSessionCfg {
    TrainSessionCfg::new(INPUT_DIM, BATCH, SEQ_LEN)
}

fn m2_cfg() -> Mamba2Config {
    Mamba2Config {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba2Config::default()
    }
}

fn mimo_cfg() -> Mamba3MimoConfig {
    Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    }
}

/// Deterministic values in [-1, 1).
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn bits<T: AsRef<[f32]>>(v: T) -> Vec<u32> {
    v.as_ref().iter().map(|x| x.to_bits()).collect()
}

/// The two families whose trainers gained the shared surface, as pairs
/// of identical trainers built fresh for each check.
#[derive(Clone, Copy, Debug)]
enum Family {
    Mamba2,
    Mimo,
}

/// A trainer of `family` with the shared surface, and its master weights
/// flattened for comparison.
enum Trainer {
    M2(Box<Mamba2GpuTrainer>),
    Mimo(Box<Mamba3MimoGpuTrainer>),
}

impl Trainer {
    /// A trainer of `family` in `dtype`; in f16 the loss scaler starts at
    /// 2^10, which keeps these windows' gradients in f16's range (at its
    /// own starting scale of 2^16 they overflow and every step skips).
    fn new(family: Family, dtype: WeightDtype) -> Self {
        let opts = GpuOptions::default().dtype(dtype);
        match family {
            Family::Mamba2 => {
                let w = Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 41);
                let mut t = Mamba2GpuTrainer::new(&w, m2_cfg(), session(), opts).unwrap();
                t.load_scaler_state(1024.0, 0).unwrap();
                Self::M2(Box::new(t))
            }
            Family::Mimo => {
                let w = Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 42);
                let mut t = Mamba3MimoGpuTrainer::new(&w, mimo_cfg(), session(), opts).unwrap();
                t.load_scaler_state(1024.0, 0).unwrap();
                Self::Mimo(Box::new(t))
            }
        }
    }

    fn as_trainer(&mut self) -> &mut dyn GpuTrainer {
        match self {
            Self::M2(t) => t.as_mut(),
            Self::Mimo(t) => t.as_mut(),
        }
    }

    /// The lower-level split these trainers keep: backward, then AdamW.
    fn backward_then_adamw(&mut self, d: &[f32]) -> bool {
        match self {
            Self::M2(t) => {
                t.backward(d, None).unwrap();
                t.apply_adamw().unwrap()
            }
            Self::Mimo(t) => {
                t.backward(d, None).unwrap();
                t.apply_adamw().unwrap()
            }
        }
    }

    /// A micro-batch joins the open window.
    fn accumulate(&mut self, d: &[f32]) {
        let m = self
            .as_trainer()
            .backward_step(d, BackwardOpts::default().with_accumulate_only(true))
            .unwrap();
        assert!(!m.optimizer_stepped);
    }

    /// The open window's gradient, read and written back through the
    /// split seam, then the optimizer tail alone.
    fn round_trip_then_apply(&mut self) -> mamba_rs::BackwardMetrics {
        match self {
            Self::M2(t) => {
                let arena = t.grad_arena().unwrap().to_cpu(&t.ctx().stream).unwrap();
                t.upload_grad_arena(&arena).unwrap();
                t.apply_step(None).unwrap()
            }
            Self::Mimo(t) => {
                let arena = t.grad_arena().unwrap().to_cpu(&t.ctx().stream).unwrap();
                t.upload_grad_arena(&arena).unwrap();
                t.apply_step(None).unwrap()
            }
        }
    }

    /// The gradient arena, flattened.
    fn grads(&self) -> Vec<f32> {
        match self {
            Self::M2(t) => flatten_m2(&t.grads().unwrap()),
            Self::Mimo(t) => flatten_mimo(&t.grads().unwrap()),
        }
    }

    fn grads_norm(&self) -> f64 {
        let flat = self.grads();
        flat.iter()
            .map(|&g| f64::from(g) * f64::from(g))
            .sum::<f64>()
            .sqrt()
    }

    fn master(&self) -> Vec<u32> {
        match self {
            Self::M2(t) => bits(flatten_m2(&t.snapshot_master().unwrap())),
            Self::Mimo(t) => bits(flatten_mimo(&t.snapshot_master().unwrap())),
        }
    }
}

/// Every tensor of the weights, in field order.
fn flatten_m2(w: &Mamba2Weights) -> Vec<f32> {
    let mut out = w.input_proj_w.clone();
    out.extend_from_slice(&w.input_proj_b);
    for l in &w.layers {
        for t in [
            &l.norm_weight,
            &l.in_proj_w,
            &l.in_proj_b,
            &l.conv_w,
            &l.conv_b,
            &l.dt_bias,
            &l.a_log,
            &l.d_param,
            &l.gate_norm_weight,
            &l.out_proj_w,
            &l.out_proj_b,
        ] {
            out.extend_from_slice(t);
        }
    }
    out.extend_from_slice(&w.norm_f_weight);
    out
}

/// Every tensor of the weights, in field order.
fn flatten_mimo(w: &Mamba3MimoWeights) -> Vec<f32> {
    let mut out = w.input_proj_w.clone();
    out.extend_from_slice(&w.input_proj_b);
    for l in &w.layers {
        for t in [
            &l.norm_weight,
            &l.in_proj_w,
            &l.dt_bias,
            &l.b_norm_weight,
            &l.c_norm_weight,
            &l.b_bias,
            &l.c_bias,
            &l.d_param,
            &l.mimo_x,
            &l.mimo_z,
            &l.mimo_o,
            &l.norm_gate_weight,
            &l.out_proj_w,
        ] {
            out.extend_from_slice(t);
        }
    }
    out.extend_from_slice(&w.norm_f_weight);
    out
}

fn window() -> (Vec<f32>, Vec<f32>) {
    (
        values(BATCH * SEQ_LEN * INPUT_DIM, 1),
        values(BATCH * SEQ_LEN * D_MODEL, 2)
            .into_iter()
            .map(|v| v * 1e-2)
            .collect(),
    )
}

fn forward(t: &mut Trainer, input: &[f32]) {
    let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
    t.as_trainer().forward(input, &mut out).unwrap();
}

/// `backward_step` with no options is the backward and the AdamW step it
/// replaces, and `step` is the forward and `backward_step`, bit for bit.
#[test]
fn backward_step_and_step_match_the_split_they_replace() {
    let (input, d) = window();
    for family in [Family::Mamba2, Family::Mimo] {
        for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
            let mut split = Trainer::new(family, dtype);
            forward(&mut split, &input);
            assert!(split.backward_then_adamw(&d));

            let mut by_step = Trainer::new(family, dtype);
            forward(&mut by_step, &input);
            let m = by_step
                .as_trainer()
                .backward_step(&d, BackwardOpts::default())
                .unwrap();
            assert!(m.optimizer_stepped && m.step == 1, "{family:?} {dtype:?}");
            assert_eq!(by_step.master(), split.master(), "{family:?} {dtype:?}");

            let mut fused = Trainer::new(family, dtype);
            let m = fused.as_trainer().step(&input, &d).unwrap();
            assert_eq!(m.step, 1, "{family:?} {dtype:?}");
            assert_eq!(fused.master(), split.master(), "{family:?} {dtype:?} step");
        }
    }
}

/// The clip reports the gradient's global norm, leaves a gradient under
/// its limit bit for bit alone, and a window whose norm passes the skip
/// threshold takes no step.
#[test]
fn clip_and_spike_skip_follow_the_gradient_norm() {
    let (input, d) = window();
    for family in [Family::Mamba2, Family::Mimo] {
        let mut probe = Trainer::new(family, WeightDtype::F32);
        forward(&mut probe, &input);
        match &mut probe {
            Trainer::M2(t) => t.backward(&d, None).unwrap(),
            Trainer::Mimo(t) => t.backward(&d, None).unwrap(),
        }
        let norm = probe.grads_norm();

        let mut plain = Trainer::new(family, WeightDtype::F32);
        forward(&mut plain, &input);
        plain
            .as_trainer()
            .backward_step(&d, BackwardOpts::default())
            .unwrap();

        let mut clipped = Trainer::new(family, WeightDtype::F32);
        forward(&mut clipped, &input);
        let m = clipped
            .as_trainer()
            .backward_step(
                &d,
                BackwardOpts::default().with_clip_max_norm(2.0 * norm as f32),
            )
            .unwrap();
        let reported = f64::from(m.grad_norm.unwrap());
        assert!(
            (reported - norm).abs() <= 1e-5 * norm,
            "{family:?}: norm {reported} against {norm}"
        );
        assert_eq!(
            clipped.master(),
            plain.master(),
            "{family:?}: under the limit"
        );

        let mut skipped = Trainer::new(family, WeightDtype::F32);
        let before = skipped.master();
        forward(&mut skipped, &input);
        let m = skipped
            .as_trainer()
            .backward_step(
                &d,
                BackwardOpts::default()
                    .with_clip_max_norm(1e9)
                    .with_step_skip_above(0.5 * norm as f32),
            )
            .unwrap();
        assert!(!m.optimizer_stepped && m.step == 0, "{family:?}: skip");
        assert_eq!(skipped.master(), before, "{family:?}: a skipped step moved");
    }
}

/// An accumulating call and an applying call on the same window step on
/// twice its gradient. Doubling is exact in floating point and the
/// backward is linear in its upstream gradient, so in f32, tf32 and bf16
/// the result has the bits of one call on the window with `d` doubled. f16
/// holds one loss scale over the window, but doubling commutes with its
/// rounding only above its subnormal range: there the window's gradient
/// agrees with the doubled call's within f16's precision.
#[test]
fn accumulation_window_steps_on_the_sum_of_its_micro_batches() {
    let (input, d) = window();
    let d2: Vec<f32> = d.iter().map(|v| 2.0 * v).collect();
    for family in [Family::Mamba2, Family::Mimo] {
        for dtype in [
            WeightDtype::F32,
            WeightDtype::Tf32,
            WeightDtype::Bf16,
            WeightDtype::F16,
        ] {
            let mut window = Trainer::new(family, dtype);
            let t = window.as_trainer();
            let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
            t.forward(&input, &mut out).unwrap();
            let a = t
                .backward_step(&d, BackwardOpts::default().with_accumulate_only(true))
                .unwrap();
            assert!(!a.optimizer_stepped && a.step == 0, "{family:?} {dtype:?}");
            t.reset_state().unwrap();
            t.forward(&input, &mut out).unwrap();
            let m = t.backward_step(&d, BackwardOpts::default()).unwrap();
            assert!(m.optimizer_stepped && m.step == 1, "{family:?} {dtype:?}");
            assert_eq!(
                m.loss_scale, a.loss_scale,
                "{family:?} {dtype:?}: the loss scale holds for the window"
            );

            let mut doubled = Trainer::new(family, dtype);
            forward(&mut doubled, &input);
            let md = doubled
                .as_trainer()
                .backward_step(&d2, BackwardOpts::default())
                .unwrap();
            assert!(md.optimizer_stepped, "{family:?} {dtype:?}: {md:?}");
            if dtype == WeightDtype::F16 {
                let (a, b) = (window.grads(), doubled.grads());
                let top = a.iter().fold(0.0f32, |t, v| t.max(v.abs()));
                let worst = a
                    .iter()
                    .zip(&b)
                    .fold(0.0f32, |t, (x, y)| t.max((x - y).abs()));
                assert!(
                    top > 0.0 && worst <= top / 1024.0,
                    "{family:?} {dtype:?}: the gradients differ by {worst:e} against {top:e}"
                );
            } else {
                assert_eq!(window.master(), doubled.master(), "{family:?} {dtype:?}");
            }
        }

        // A clip belongs on the applying call.
        let mut f32_t = Trainer::new(family, WeightDtype::F32);
        forward(&mut f32_t, &input);
        let refused = f32_t.as_trainer().backward_step(
            &d,
            BackwardOpts::default()
                .with_accumulate_only(true)
                .with_clip_max_norm(1.0),
        );
        assert!(
            refused.is_err(),
            "{family:?}: a clip on an accumulating call"
        );
    }
}

/// The split seam: a window whose last micro-batch only accumulates and
/// whose optimizer tail runs alone through `apply_step` steps on the bits
/// an applying `backward_step` gives, in every precision; its arena reads
/// out and uploads back unchanged on the way. Without an open window
/// there is no arena and nothing to apply.
#[test]
fn apply_step_closes_a_window_as_an_applying_backward_does() {
    let (input, d) = window();
    for family in [Family::Mamba2, Family::Mimo] {
        for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
            let mut applying = Trainer::new(family, dtype);
            forward(&mut applying, &input);
            applying.accumulate(&d);
            forward(&mut applying, &input);
            let a = applying
                .as_trainer()
                .backward_step(&d, BackwardOpts::default())
                .unwrap();

            let mut split = Trainer::new(family, dtype);
            forward(&mut split, &input);
            split.accumulate(&d);
            forward(&mut split, &input);
            split.accumulate(&d);
            let m = split.round_trip_then_apply();
            assert!(m.optimizer_stepped && m.step == 1, "{family:?} {dtype:?}");
            assert_eq!(
                (m.loss_scale, m.overflow_skipped),
                (a.loss_scale, a.overflow_skipped),
                "{family:?} {dtype:?}"
            );
            assert_eq!(split.master(), applying.master(), "{family:?} {dtype:?}");
            let closed = match &mut split {
                Trainer::M2(t) => (t.grad_arena().is_err(), t.apply_step(None).is_err()),
                Trainer::Mimo(t) => (t.grad_arena().is_err(), t.apply_step(None).is_err()),
            };
            assert_eq!(
                closed,
                (true, true),
                "{family:?} {dtype:?}: a closed window"
            );
        }
    }
}

/// A few steps toward a fixed target lower the loss in every family,
/// driven only through the trait, and the learning rate round-trips.
/// The spike skip measures the gradient norm itself when no clip is set:
/// a window above the threshold takes no step in any family, and the
/// metrics report the norm that decided it.
#[test]
fn the_spike_skip_needs_no_clip_in_every_family() {
    let (m1_cfg, siso_cfg) = (m1_cfg(), siso_cfg());
    let (input, d) = window();
    let opts = GpuOptions::default().dtype(WeightDtype::F32);
    let s = session();
    let trainers: Vec<(&str, Box<dyn GpuTrainer>)> = vec![
        (
            "Mamba-1",
            Box::new(
                MambaGpuTrainer::new(&MambaWeights::init(&m1_cfg, INPUT_DIM, 61), m1_cfg, s, opts)
                    .unwrap(),
            ),
        ),
        (
            "Mamba-2",
            Box::new(
                Mamba2GpuTrainer::new(
                    &Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 62),
                    m2_cfg(),
                    s,
                    opts,
                )
                .unwrap(),
            ),
        ),
        (
            "Mamba-3 SISO",
            Box::new(
                Mamba3GpuTrainer::new(
                    &Mamba3Weights::init(&siso_cfg, INPUT_DIM, 63),
                    siso_cfg,
                    s,
                    opts,
                )
                .unwrap(),
            ),
        ),
        (
            "Mamba-3 MIMO",
            Box::new(
                Mamba3MimoGpuTrainer::new(
                    &Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 64),
                    mimo_cfg(),
                    s,
                    opts,
                )
                .unwrap(),
            ),
        ),
    ];
    for (name, mut t) in trainers {
        let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
        t.forward(&input, &mut out).unwrap();
        let m = t
            .backward_step(&d, BackwardOpts::default().with_step_skip_above(1e-12))
            .unwrap();
        assert!(
            !m.optimizer_stepped && m.step == 0,
            "{name}: a window above the skip threshold stepped"
        );
        let norm = m.grad_norm.expect("the skip reports the norm it measured");
        assert!(norm.is_finite() && norm > 0.0, "{name}: norm {norm}");
    }
}

#[test]
fn every_family_trains_through_the_trait() {
    let (m1_cfg, siso_cfg) = (m1_cfg(), siso_cfg());
    let s = TrainSessionCfg {
        lr: 3e-3,
        ..session()
    };
    for dtype in [
        WeightDtype::F32,
        WeightDtype::Tf32,
        WeightDtype::Bf16,
        WeightDtype::F16,
    ] {
        let opts = GpuOptions::default().dtype(dtype);
        let mut trainers: Vec<(&str, Box<dyn GpuTrainer>)> = vec![
            (
                "Mamba-1",
                Box::new(
                    MambaGpuTrainer::new(
                        &MambaWeights::init(&m1_cfg, INPUT_DIM, 51),
                        m1_cfg,
                        s,
                        opts,
                    )
                    .unwrap(),
                ),
            ),
            (
                "Mamba-2",
                Box::new(
                    Mamba2GpuTrainer::new(
                        &Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 52),
                        m2_cfg(),
                        s,
                        opts,
                    )
                    .unwrap(),
                ),
            ),
            (
                "Mamba-3 SISO",
                Box::new(
                    Mamba3GpuTrainer::new(
                        &Mamba3Weights::init(&siso_cfg, INPUT_DIM, 53),
                        siso_cfg,
                        s,
                        opts,
                    )
                    .unwrap(),
                ),
            ),
            (
                "Mamba-3 MIMO",
                Box::new(
                    Mamba3MimoGpuTrainer::new(
                        &Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 54),
                        mimo_cfg(),
                        s,
                        opts,
                    )
                    .unwrap(),
                ),
            ),
        ];
        let input = values(BATCH * SEQ_LEN * INPUT_DIM, 7);
        let target = values(BATCH * SEQ_LEN * D_MODEL, 8);
        for (name, t) in &mut trainers {
            assert_eq!(
                (t.batch(), t.seq_len(), t.input_dim(), t.d_model()),
                (BATCH, SEQ_LEN, INPUT_DIM, D_MODEL),
                "{name}"
            );
            assert_eq!(t.lr(), 3e-3, "{name}");
            t.set_lr(2e-3).unwrap();
            assert_eq!(t.lr(), 2e-3, "{name}");
            assert!(t.set_lr(f32::NAN).is_err(), "{name}");
            let n = (BATCH * SEQ_LEN * D_MODEL) as f32;
            let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
            let mut losses = Vec::new();
            for _ in 0..6 {
                t.reset_state().unwrap();
                t.forward(&input, &mut out).unwrap();
                let loss: f32 = out
                    .iter()
                    .zip(&target)
                    .map(|(o, y)| (o - y) * (o - y))
                    .sum::<f32>()
                    / n;
                let d: Vec<f32> = out
                    .iter()
                    .zip(&target)
                    .map(|(o, y)| 2.0 * (o - y) / n)
                    .collect();
                t.backward_step(&d, BackwardOpts::default()).unwrap();
                losses.push(loss);
            }
            eprintln!(
                "{name} {dtype:?}: loss {:.5} -> {:.5}",
                losses[0], losses[5]
            );
            assert!(losses[5] < losses[0], "{name} {dtype:?}: {losses:?}");
        }
    }
}

fn m1_cfg() -> MambaConfig {
    MambaConfig {
        d_model: D_MODEL,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
    }
}

fn siso_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: D_MODEL,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        ..Mamba3Config::default()
    }
}

/// One trainer of each family behind `AnyGpuTrainer`, in `dtype`.
fn any_trainers(dtype: WeightDtype) -> Vec<AnyGpuTrainer> {
    let (s, opts) = (session(), GpuOptions::default().dtype(dtype));
    let mut out: Vec<AnyGpuTrainer> = vec![
        MambaGpuTrainer::new(
            &MambaWeights::init(&m1_cfg(), INPUT_DIM, 61),
            m1_cfg(),
            s,
            opts,
        )
        .unwrap()
        .into(),
        Mamba2GpuTrainer::new(
            &Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 62),
            m2_cfg(),
            s,
            opts,
        )
        .unwrap()
        .into(),
        Mamba3GpuTrainer::new(
            &Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 63),
            siso_cfg(),
            s,
            opts,
        )
        .unwrap()
        .into(),
        Mamba3MimoGpuTrainer::new(
            &Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 64),
            mimo_cfg(),
            s,
            opts,
        )
        .unwrap()
        .into(),
    ];
    for t in &mut out {
        t.load_scaler_state(1024.0, 0).unwrap();
    }
    out
}

/// The shared surface carries the graphs, the split seam and the
/// data-parallel entry in every family through `AnyGpuTrainer`: a captured
/// step replays, an accumulated window's arena reads out and uploads back
/// into an `apply_step_full` that steps, a single-rank
/// `backward_step_dist` is the plain applying call bit for bit, and a loss
/// scale that is not finite and positive is refused.
#[test]
fn the_shared_surface_carries_graphs_seam_and_dist_in_every_family() {
    let (input, d) = window();
    let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
    let families = [
        ModelFamily::Mamba1,
        ModelFamily::Mamba2,
        ModelFamily::Mamba3,
        ModelFamily::Mamba3Mimo,
    ];
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        let mut graphs = any_trainers(dtype);
        let mut plain = any_trainers(dtype);
        let mut dist = any_trainers(dtype);
        let single = DistContext::single(0, 0);
        for (i, family) in families.iter().enumerate() {
            let label = format!("{family:?} {dtype:?}");
            let t = &mut graphs[i];
            assert_eq!(t.family(), *family, "{label}");
            assert!(
                t.load_scaler_state(f32::NAN, 0).is_err(),
                "{label}: NaN scale"
            );
            assert!(t.load_scaler_state(0.0, 0).is_err(), "{label}: zero scale");
            t.step(&input, &d).unwrap();
            t.capture_graph().unwrap();
            assert!(t.has_graph(), "{label}");
            assert!(
                t.step(&input, &d).unwrap().graph_replayed,
                "{label}: replay"
            );
            t.drop_graph();
            assert!(!t.has_graph(), "{label}");
            assert!(
                !t.step(&input, &d).unwrap().graph_replayed,
                "{label}: eager"
            );

            let t = &mut plain[i];
            t.forward(&input, &mut out).unwrap();
            t.backward_step(&d, BackwardOpts::default().with_accumulate_only(true))
                .unwrap();
            let arena = t.download_grad_arena().unwrap();
            assert!(arena.iter().any(|&g| g != 0.0), "{label}: an empty arena");
            t.upload_grad_arena(&arena).unwrap();
            let m = t.apply_step_full(None, None, None).unwrap();
            assert!(m.optimizer_stepped && m.step == 1, "{label}: {m:?}");

            let t = &mut dist[i];
            t.forward(&input, &mut out).unwrap();
            let m = t
                .backward_step_dist(&d, BackwardOpts::default(), &single)
                .unwrap();
            assert!(m.optimizer_stepped && m.step == 1, "{label}: {m:?}");
        }
        // One applying step from the same start: the seam and the
        // single-rank path give the same weights.
        for (i, family) in families.iter().enumerate() {
            let a = plain[i].download_grad_arena();
            assert!(a.is_err() || matches!(family, ModelFamily::Mamba1 | ModelFamily::Mamba3));
            let (p, q) = (&plain[i], &dist[i]);
            assert_eq!(
                snapshot_bits(p),
                snapshot_bits(q),
                "{family:?} {dtype:?}: the seam against the single-rank path"
            );
        }
    }
}

/// The step options reach the fused step in every family and precision,
/// eager and captured: three steps with the control clip and the global
/// clip give the weights of the split call with the same options bit for
/// bit, eager and replayed alike, and report the norm. A spike threshold
/// below every norm skips the step on both roads, the weights and the
/// step count where they were. Accumulation is refused as a step option,
/// and so is a change under a captured graph until it is dropped.
#[test]
fn the_step_options_reach_the_eager_and_the_captured_step_in_every_family() {
    let (input, d) = window();
    let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
    let opts = BackwardOpts::default()
        .with_clip_max_norm(0.05)
        .with_control_clip_max_norm(0.01);
    let spike = opts.with_step_skip_above(1e-12);
    for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
        let mut eager = any_trainers(dtype);
        let mut split = any_trainers(dtype);
        let mut graph = any_trainers(dtype);
        for i in 0..eager.len() {
            let label = format!("{:?} {dtype:?}", eager[i].family());
            assert!(
                eager[i]
                    .set_step_opts(opts.with_accumulate_only(true))
                    .is_err(),
                "{label}: accumulation as a step option"
            );
            eager[i].set_step_opts(opts).unwrap();
            assert_eq!(eager[i].step_opts(), opts, "{label}");
            graph[i].set_step_opts(opts).unwrap();
            for k in 0..3 {
                let m = eager[i].step(&input, &d).unwrap();
                assert!(
                    m.optimizer_stepped && m.grad_norm.is_some_and(f32::is_finite),
                    "{label} step {k}: {m:?}"
                );
                split[i].forward(&input, &mut out).unwrap();
                split[i].backward_step(&d, opts).unwrap();
                if k == 0 {
                    graph[i].step(&input, &d).unwrap();
                    graph[i].capture_graph().unwrap();
                } else {
                    let m = graph[i].step(&input, &d).unwrap();
                    assert!(m.graph_replayed && m.optimizer_stepped, "{label}: {m:?}");
                }
            }
            let want = snapshot_bits(&split[i]);
            assert_eq!(snapshot_bits(&eager[i]), want, "{label}: eager step");
            assert_eq!(snapshot_bits(&graph[i]), want, "{label}: captured step");

            assert!(
                graph[i].set_step_opts(spike).is_err(),
                "{label}: a change under a captured graph"
            );
            graph[i].set_step_opts(opts).unwrap();
            graph[i].drop_graph();
            graph[i].set_step_opts(spike).unwrap();
            graph[i].step(&input, &d).unwrap();
            graph[i].capture_graph().unwrap();
            let before = snapshot_bits(&graph[i]);
            let steps = graph[i].optimizer_state().unwrap().step;
            let m = graph[i].step(&input, &d).unwrap();
            assert!(
                m.graph_replayed && !m.optimizer_stepped && m.step == steps,
                "{label}: the spike replay {m:?}"
            );
            assert_eq!(
                snapshot_bits(&graph[i]),
                before,
                "{label}: a skipped replay"
            );
            eager[i].set_step_opts(spike).unwrap();
            let before = snapshot_bits(&eager[i]);
            let m = eager[i].step(&input, &d).unwrap();
            assert!(
                !m.optimizer_stepped && m.step == 3,
                "{label}: the spike step {m:?}"
            );
            assert_eq!(snapshot_bits(&eager[i]), before, "{label}: a skipped step");
        }
    }
}

/// A window of micro-batches trains through `accumulate` and
/// `apply_window` in every family and precision, eager and captured, with
/// the bits of the split calls it replaces: a fused step, then two windows
/// of two micro-batches with both clips, and the replays report it. In f16
/// a micro-batch that overflows skips its window on both roads, the
/// weights and the step count where they were and the scale backed off.
#[test]
fn a_window_trains_through_accumulate_and_apply_in_every_family() {
    let (input, d) = window();
    let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
    let opts = BackwardOpts::default()
        .with_clip_max_norm(0.05)
        .with_control_clip_max_norm(0.01);
    let acc = BackwardOpts::default().with_accumulate_only(true);
    let batches: Vec<(Vec<f32>, Vec<f32>)> = (0..4u32)
        .map(|k| {
            let x = values(input.len(), 300 + k);
            let g: Vec<f32> = values(d.len(), 400 + k).iter().map(|v| v * 1e-2).collect();
            (x, g)
        })
        .collect();
    for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
        let mut eager = any_trainers(dtype);
        let mut split = any_trainers(dtype);
        let mut graph = any_trainers(dtype);
        for i in 0..eager.len() {
            let label = format!("{:?} {dtype:?}", eager[i].family());
            assert!(
                eager[i].apply_window().is_err(),
                "{label}: an apply without a window"
            );
            eager[i].set_step_opts(opts).unwrap();
            graph[i].set_step_opts(opts).unwrap();
            eager[i].step(&input, &d).unwrap();
            graph[i].step(&input, &d).unwrap();
            split[i].forward(&input, &mut out).unwrap();
            split[i].backward_step(&d, opts).unwrap();
            graph[i].capture_graph().unwrap();
            for w in batches.chunks(2) {
                for (x, g) in w {
                    let m = eager[i].accumulate(x, g).unwrap();
                    assert!(!m.optimizer_stepped, "{label}: {m:?}");
                    graph[i].accumulate(x, g).unwrap();
                    split[i].forward(x, &mut out).unwrap();
                    split[i].backward_step(g, acc).unwrap();
                }
                let m = eager[i].apply_window().unwrap();
                assert!(
                    m.optimizer_stepped && !m.graph_replayed && m.grad_norm.is_some(),
                    "{label}: the eager apply {m:?}"
                );
                let m = graph[i].apply_window().unwrap();
                assert!(
                    m.optimizer_stepped && m.graph_replayed && m.grad_norm.is_some(),
                    "{label}: the replayed apply {m:?}"
                );
                split[i]
                    .apply_step_full(
                        opts.clip_max_norm,
                        opts.step_skip_above,
                        opts.control_clip_max_norm,
                    )
                    .unwrap();
            }
            let want = snapshot_bits(&split[i]);
            assert_eq!(snapshot_bits(&eager[i]), want, "{label}: eager window");
            assert_eq!(snapshot_bits(&graph[i]), want, "{label}: captured window");
            if dtype != WeightDtype::F16 {
                continue;
            }
            for t in [&mut eager[i], &mut graph[i]] {
                t.load_scaler_state(3.0e38, 0).unwrap();
                let (before, steps) = (snapshot_bits(t), t.optimizer_state().unwrap().step);
                let (x, g) = &batches[0];
                t.accumulate(x, g).unwrap();
                t.accumulate(x, g).unwrap();
                let m = t.apply_window().unwrap();
                assert!(
                    m.overflow_skipped == Some(true) && !m.optimizer_stepped && m.step == steps,
                    "{label}: an overflowed window {m:?}"
                );
                assert_eq!(snapshot_bits(t), before, "{label}: a skipped window");
                assert!(
                    t.scaler_state().unwrap().0 < 3.0e38,
                    "{label}: the scale did not back off"
                );
            }
        }
    }
}

/// A fused step with an accumulation window open adds its micro-batch to
/// the window and applies it, in every family and precision: the bits of
/// an accumulating backward followed by an applying one.
#[test]
fn a_fused_step_on_an_open_window_applies_it_in_every_family() {
    let (input, d) = window();
    let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
    let acc = BackwardOpts::default().with_accumulate_only(true);
    let second = values(input.len(), 900);
    for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
        let mut fused = any_trainers(dtype);
        let mut split = any_trainers(dtype);
        for i in 0..fused.len() {
            let label = format!("{:?} {dtype:?}", fused[i].family());
            fused[i].accumulate(&input, &d).unwrap();
            split[i].forward(&input, &mut out).unwrap();
            split[i].backward_step(&d, acc).unwrap();
            let m = fused[i].step(&second, &d).unwrap();
            assert!(m.optimizer_stepped && m.step == 1, "{label}: {m:?}");
            split[i].forward(&second, &mut out).unwrap();
            split[i].backward_step(&d, BackwardOpts::default()).unwrap();
            assert_eq!(
                snapshot_bits(&fused[i]),
                snapshot_bits(&split[i]),
                "{label}"
            );
        }
    }
}

/// Write the trainer's master weights to its family's checkpoint, and build
/// a trainer of the same family from that file.
fn reload_through_checkpoint(t: &AnyGpuTrainer, path: &Path, opts: GpuOptions) -> AnyGpuTrainer {
    let s = session();
    match t {
        AnyGpuTrainer::M1(t) => {
            let w = t.snapshot_master().unwrap();
            mamba_rs::mamba1::serialize::save(path, &w, &m1_cfg(), INPUT_DIM).unwrap();
            let (w, cfg, input_dim) = mamba_rs::mamba1::serialize::load(path).unwrap();
            assert_eq!(input_dim, INPUT_DIM);
            MambaGpuTrainer::new(&w, cfg, s, opts).unwrap().into()
        }
        AnyGpuTrainer::M2(t) => {
            let w = t.snapshot_master().unwrap();
            mamba_rs::mamba2::serialize::save(path, &w, &m2_cfg(), INPUT_DIM).unwrap();
            let (w, cfg, input_dim) = mamba_rs::mamba2::serialize::load(path).unwrap();
            assert_eq!(input_dim, INPUT_DIM);
            Mamba2GpuTrainer::new(&w, cfg, s, opts).unwrap().into()
        }
        AnyGpuTrainer::M3(t) => {
            let w = t.snapshot_master().unwrap();
            mamba_rs::mamba3_siso::serialize::save(path, &w, &siso_cfg(), INPUT_DIM).unwrap();
            let (w, cfg, input_dim) = mamba_rs::mamba3_siso::serialize::load(path).unwrap();
            assert_eq!(input_dim, INPUT_DIM);
            Mamba3GpuTrainer::new(&w, cfg, s, opts).unwrap().into()
        }
        AnyGpuTrainer::M3Mimo(t) => {
            let w = t.snapshot_master().unwrap();
            mamba_rs::mamba3_mimo::serialize::save(path, &w, &mimo_cfg(), INPUT_DIM).unwrap();
            let (w, cfg, input_dim) = mamba_rs::mamba3_mimo::serialize::load(path).unwrap();
            assert_eq!(input_dim, INPUT_DIM);
            Mamba3MimoGpuTrainer::new(&w, cfg, s, opts).unwrap().into()
        }
    }
}

/// Every family resumes bit for bit through files: the weights through
/// the family's checkpoint, the optimizer, the loss scale and the carried
/// recurrence through the trainer-state file. Two steps, both files
/// written, a fresh trainer built from them, two more steps: the weights
/// and the state of four unbroken steps. Every step steps, in f16 too, so
/// the comparison bites; a state of another family is refused.
#[test]
fn every_family_resumes_through_its_checkpoint_and_trainer_state_files() {
    let (input, d) = window();
    let dir = std::env::temp_dir();
    for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
        let opts = GpuOptions::default().dtype(dtype);
        for (mut a, mut b) in any_trainers(dtype).into_iter().zip(any_trainers(dtype)) {
            let family = a.family();
            let label = format!("{family:?} {dtype:?}");
            let step = |t: &mut AnyGpuTrainer| {
                let m = t.step(&input, &d).unwrap();
                assert_ne!(m.overflow_skipped, Some(true), "{label}: {m:?}");
            };
            for _ in 0..4 {
                step(&mut a);
            }
            for _ in 0..2 {
                step(&mut b);
            }
            let name = format!(
                "mamba_rs_resume_{}_{family:?}_{dtype:?}",
                std::process::id()
            );
            let weights = dir.join(format!("{name}_weights.safetensors"));
            let state_file = dir.join(format!("{name}_state.safetensors"));
            b.trainer_state().unwrap().save(&state_file).unwrap();
            let mut c = reload_through_checkpoint(&b, &weights, opts);
            let state = TrainerState::load(&state_file);
            std::fs::remove_file(&weights).ok();
            std::fs::remove_file(&state_file).ok();
            let state = state.unwrap();
            assert_eq!(state, b.trainer_state().unwrap(), "{label}: the file");
            assert_eq!(state.scaler.is_some(), dtype == WeightDtype::F16, "{label}");

            let mut foreign = state.clone();
            foreign.family = if family == ModelFamily::Mamba1 {
                ModelFamily::Mamba2
            } else {
                ModelFamily::Mamba1
            };
            assert!(
                c.load_trainer_state(&foreign).is_err(),
                "{label}: another family's state"
            );
            c.load_trainer_state(&state).unwrap();
            for _ in 0..2 {
                step(&mut c);
            }
            assert_eq!(snapshot_bits(&c), snapshot_bits(&a), "{label}: the weights");
            assert_eq!(
                c.trainer_state().unwrap(),
                a.trainer_state().unwrap(),
                "{label}: the optimizer, scale and recurrence"
            );
        }
    }
}

/// The master weights of any family's trainer, as bits.
fn snapshot_bits(t: &AnyGpuTrainer) -> Vec<u32> {
    match t {
        AnyGpuTrainer::M1(t) => {
            let w = t.snapshot_master().unwrap();
            let mut v: Vec<f32> = w.input_proj_w.clone();
            v.extend_from_slice(&w.input_proj_b);
            for l in &w.layers {
                for x in [
                    &l.norm_weight,
                    &l.in_proj_w,
                    &l.conv1d_weight,
                    &l.conv1d_bias,
                    &l.x_proj_w,
                    &l.dt_proj_w,
                    &l.dt_proj_b,
                    &l.a_log,
                    &l.d_param,
                    &l.out_proj_w,
                ] {
                    v.extend_from_slice(x);
                }
            }
            v.extend_from_slice(&w.norm_f_weight);
            bits(v)
        }
        AnyGpuTrainer::M2(t) => bits(flatten_m2(&t.snapshot_master().unwrap())),
        AnyGpuTrainer::M3(t) => {
            let w = t.snapshot_master().unwrap();
            let mut v: Vec<f32> = w.input_proj_w.clone();
            v.extend_from_slice(&w.input_proj_b);
            for l in &w.layers {
                for x in [
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
                    v.extend_from_slice(x);
                }
            }
            v.extend_from_slice(&w.norm_f_weight);
            bits(v)
        }
        AnyGpuTrainer::M3Mimo(t) => bits(flatten_mimo(&t.snapshot_master().unwrap())),
    }
}

/// The input gradient of every family: in f32 it matches central
/// differences of `sum(out * d)` over the window's input (each forward
/// from a zero state, the accumulating backward leaving the weights
/// alone), and the half precisions give it within their rounding.
#[test]
fn every_family_returns_the_input_gradient() {
    let (input, d) = window();
    let n_out = BATCH * SEQ_LEN * D_MODEL;
    let accumulating = BackwardOpts::default().with_accumulate_only(true);
    let loss = |t: &mut AnyGpuTrainer, x: &[f32]| -> f64 {
        let mut out = vec![0.0f32; n_out];
        t.reset_state().unwrap();
        t.forward(x, &mut out).unwrap();
        out.iter()
            .zip(&d)
            .map(|(o, g)| f64::from(*o) * f64::from(*g))
            .sum()
    };
    let grads = |dtype: WeightDtype| -> Vec<Vec<f32>> {
        any_trainers(dtype)
            .iter_mut()
            .map(|t| {
                let mut out = vec![0.0f32; n_out];
                let mut d_input = vec![0.0f32; input.len()];
                t.reset_state().unwrap();
                t.forward(&input, &mut out).unwrap();
                t.backward_step_input_grad(&d, accumulating, &mut d_input)
                    .unwrap();
                d_input
            })
            .collect()
    };
    let exact = grads(WeightDtype::F32);
    let mut fd_trainers = any_trainers(WeightDtype::F32);
    for (t, g) in fd_trainers.iter_mut().zip(&exact) {
        let family = t.family();
        // The four largest entries, where a wrong gradient shows.
        let mut order: Vec<usize> = (0..g.len()).collect();
        order.sort_by(|&a, &b| g[b].abs().total_cmp(&g[a].abs()));
        let h = 1e-2f32;
        for &k in &order[..4] {
            let mut x = input.clone();
            x[k] += h;
            let up = loss(t, &x);
            x[k] -= 2.0 * h;
            let down = loss(t, &x);
            let fd = (up - down) / (2.0 * f64::from(h));
            let got = f64::from(g[k]);
            assert!(
                (fd - got).abs() <= 2e-3 + 5e-2 * got.abs(),
                "{family:?}: d input[{k}] is {got}, central differences give {fd}"
            );
        }
    }
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for (family, (half, f32_g)) in grads(dtype).iter().zip(&exact).enumerate() {
            let norm = f32_g
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt();
            let diff = half
                .iter()
                .zip(f32_g)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                diff <= 0.05 * norm,
                "family {family} {dtype:?}: input gradient off f32 by {diff} of {norm}"
            );
        }
    }
}
