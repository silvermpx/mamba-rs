//! The GPU trainer surface every family shares.
//!
//! For Mamba-2 and MIMO, `backward_step` is the backward and the AdamW
//! step it replaces, `step` is forward plus `backward_step`, the clip
//! reports the gradient's norm and leaves a gradient under the limit
//! alone, a spiking window is dropped whole, and an accumulation window
//! steps on the sum of its micro-batches. Every family trains through the
//! `GpuTrainer` trait.
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::{
    BackwardOpts, GpuOptions, GpuTrainer, Mamba2Config, Mamba2GpuTrainer, Mamba2Weights,
    Mamba3Config, Mamba3GpuTrainer, Mamba3MimoConfig, Mamba3MimoGpuTrainer, Mamba3MimoWeights,
    Mamba3Weights, MambaConfig, MambaGpuTrainer, MambaWeights, TrainSessionCfg, WeightDtype,
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
    fn new(family: Family, dtype: WeightDtype) -> Self {
        let opts = GpuOptions::default().dtype(dtype);
        match family {
            Family::Mamba2 => {
                let w = Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 41);
                Self::M2(Box::new(
                    Mamba2GpuTrainer::new(&w, m2_cfg(), session(), opts).unwrap(),
                ))
            }
            Family::Mimo => {
                let w = Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 42);
                Self::Mimo(Box::new(
                    Mamba3MimoGpuTrainer::new(&w, mimo_cfg(), session(), opts).unwrap(),
                ))
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

    fn grads_norm(&self) -> f64 {
        let flat: Vec<f32> = match self {
            Self::M2(t) => flatten_m2(&t.grads().unwrap()),
            Self::Mimo(t) => flatten_mimo(&t.grads().unwrap()),
        };
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
/// backward is linear in its upstream gradient, so the result has the bits
/// of one call on the window with `d` doubled.
#[test]
fn accumulation_window_steps_on_the_sum_of_its_micro_batches() {
    let (input, d) = window();
    let d2: Vec<f32> = d.iter().map(|v| 2.0 * v).collect();
    for family in [Family::Mamba2, Family::Mimo] {
        let mut window = Trainer::new(family, WeightDtype::F32);
        let t = window.as_trainer();
        let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
        t.forward(&input, &mut out).unwrap();
        let m = t
            .backward_step(&d, BackwardOpts::default().with_accumulate_only(true))
            .unwrap();
        assert!(!m.optimizer_stepped && m.step == 0, "{family:?}");
        t.reset_state().unwrap();
        t.forward(&input, &mut out).unwrap();
        let m = t.backward_step(&d, BackwardOpts::default()).unwrap();
        assert!(m.optimizer_stepped && m.step == 1, "{family:?}");

        let mut doubled = Trainer::new(family, WeightDtype::F32);
        forward(&mut doubled, &input);
        doubled
            .as_trainer()
            .backward_step(&d2, BackwardOpts::default())
            .unwrap();
        assert_eq!(window.master(), doubled.master(), "{family:?}");

        // f16 has no accumulation window, and a clip belongs on the
        // applying call.
        let mut half = Trainer::new(family, WeightDtype::F16);
        forward(&mut half, &input);
        let refused = half
            .as_trainer()
            .backward_step(&d, BackwardOpts::default().with_accumulate_only(true));
        assert!(refused.is_err(), "{family:?}: f16 accumulated");
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

/// A few steps toward a fixed target lower the loss in every family,
/// driven only through the trait, and the learning rate round-trips.
#[test]
fn every_family_trains_through_the_trait() {
    let m1_cfg = MambaConfig {
        d_model: D_MODEL,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    };
    let siso_cfg = Mamba3Config {
        d_model: D_MODEL,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        ..Mamba3Config::default()
    };
    let opts = GpuOptions::default();
    let s = TrainSessionCfg {
        lr: 3e-3,
        ..session()
    };
    let mut trainers: Vec<(&str, Box<dyn GpuTrainer>)> = vec![
        (
            "Mamba-1",
            Box::new(
                MambaGpuTrainer::new(&MambaWeights::init(&m1_cfg, INPUT_DIM, 51), m1_cfg, s, opts)
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
        eprintln!("{name}: loss {:.5} -> {:.5}", losses[0], losses[5]);
        assert!(losses[5] < losses[0], "{name}: {losses:?}");
    }
}
