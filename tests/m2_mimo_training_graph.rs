//! The Mamba-2 and MIMO training step captured as one CUDA graph replays
//! with the bits of the eager step, in every storage precision: the same
//! master weights, step counts and, in f16, loss scales, through a
//! learning-rate change between replays and an overflow the captured AdamW
//! skips on the device. The replay refuses AdamW settings the capture
//! baked in until the graph is captured again, and a step with an open
//! accumulation window runs eagerly to fold the window in.
#![cfg(feature = "cuda")]

use mamba_rs::{
    BackwardOpts, GpuOptions, Mamba2Config, Mamba2GpuTrainer, Mamba2Weights, Mamba3MimoConfig,
    Mamba3MimoGpuTrainer, Mamba3MimoWeights, StepMetrics, TrainSessionCfg, WeightDtype,
};

const BATCH: usize = 2;
const SEQ_LEN: usize = 48;
const D_MODEL: usize = 32;
const INPUT_DIM: usize = 24;
/// The f16 loss scale the trainers start from: at the scaler's own 2^16
/// these windows' gradients pass f16's range, so only the window made to
/// overflow does. No effect in the other precisions.
const F16_START_SCALE: f32 = 1024.0;

fn session() -> TrainSessionCfg {
    TrainSessionCfg::new(INPUT_DIM, BATCH, SEQ_LEN)
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

/// Window `k`'s input and upstream gradient; `blow` makes the gradient
/// overflow any f16 loss scale.
fn window(k: u32, blow: bool) -> (Vec<f32>, Vec<f32>) {
    let mut d: Vec<f32> = values(BATCH * SEQ_LEN * D_MODEL, 2 * k + 2)
        .into_iter()
        .map(|v| v * 1e-2)
        .collect();
    if blow {
        d[0] = f32::MAX;
    }
    (values(BATCH * SEQ_LEN * INPUT_DIM, 2 * k + 1), d)
}

/// The surface of one family's trainer this test drives.
trait Trainer {
    fn step(&mut self, input: &[f32], d: &[f32]) -> Result<StepMetrics, String>;
    fn capture(&mut self) -> Result<(), String>;
    fn drop_graph(&mut self);
    fn master(&self) -> Vec<u32>;
    fn scaler(&self) -> Option<(f32, u32)>;
    fn set_lr(&mut self, lr: f32);
    fn set_beta1(&mut self, beta1: f32);
    fn accumulate(&mut self, input: &[f32], d: &[f32]);
}

fn bits(v: &[f32]) -> impl Iterator<Item = u32> + '_ {
    v.iter().map(|x| x.to_bits())
}

impl Trainer for Mamba2GpuTrainer {
    fn step(&mut self, input: &[f32], d: &[f32]) -> Result<StepMetrics, String> {
        Mamba2GpuTrainer::step(self, input, d)
    }
    fn capture(&mut self) -> Result<(), String> {
        self.capture_graph()
    }
    fn drop_graph(&mut self) {
        Mamba2GpuTrainer::drop_graph(self);
    }
    fn master(&self) -> Vec<u32> {
        let w = self.snapshot_master().unwrap();
        let mut out: Vec<u32> = bits(&w.input_proj_w).chain(bits(&w.input_proj_b)).collect();
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
                out.extend(bits(t));
            }
        }
        out.extend(bits(&w.norm_f_weight));
        out
    }
    fn scaler(&self) -> Option<(f32, u32)> {
        self.scaler_state()
    }
    fn set_lr(&mut self, lr: f32) {
        Mamba2GpuTrainer::set_lr(self, lr).unwrap();
    }
    fn set_beta1(&mut self, beta1: f32) {
        self.adamw_mut().beta1 = beta1;
    }
    fn accumulate(&mut self, input: &[f32], d: &[f32]) {
        let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
        self.forward(input, &mut out).unwrap();
        self.backward_step(d, BackwardOpts::default().with_accumulate_only(true))
            .unwrap();
    }
}

impl Trainer for Mamba3MimoGpuTrainer {
    fn step(&mut self, input: &[f32], d: &[f32]) -> Result<StepMetrics, String> {
        Mamba3MimoGpuTrainer::step(self, input, d)
    }
    fn capture(&mut self) -> Result<(), String> {
        self.capture_graph()
    }
    fn drop_graph(&mut self) {
        Mamba3MimoGpuTrainer::drop_graph(self);
    }
    fn master(&self) -> Vec<u32> {
        let w = self.snapshot_master().unwrap();
        let mut out: Vec<u32> = bits(&w.input_proj_w).chain(bits(&w.input_proj_b)).collect();
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
                out.extend(bits(t));
            }
        }
        out.extend(bits(&w.norm_f_weight));
        out
    }
    fn scaler(&self) -> Option<(f32, u32)> {
        self.scaler_state()
    }
    fn set_lr(&mut self, lr: f32) {
        Mamba3MimoGpuTrainer::set_lr(self, lr).unwrap();
    }
    fn set_beta1(&mut self, beta1: f32) {
        self.adamw_mut().beta1 = beta1;
    }
    fn accumulate(&mut self, input: &[f32], d: &[f32]) {
        let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
        self.forward(input, &mut out).unwrap();
        self.backward_step(d, BackwardOpts::default().with_accumulate_only(true))
            .unwrap();
    }
}

fn m2(dtype: WeightDtype) -> Box<dyn Trainer> {
    let cfg = Mamba2Config {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba2Config::default()
    };
    let w = Mamba2Weights::init(&cfg, INPUT_DIM, 41);
    let mut t =
        Mamba2GpuTrainer::new(&w, cfg, session(), GpuOptions::default().dtype(dtype)).unwrap();
    t.load_scaler_state(F16_START_SCALE, 0).unwrap();
    Box::new(t)
}

fn mimo(dtype: WeightDtype) -> Box<dyn Trainer> {
    let cfg = Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    };
    let w = Mamba3MimoWeights::init(&cfg, INPUT_DIM, 42);
    let mut t =
        Mamba3MimoGpuTrainer::new(&w, cfg, session(), GpuOptions::default().dtype(dtype)).unwrap();
    t.load_scaler_state(F16_START_SCALE, 0).unwrap();
    Box::new(t)
}

type Family = (&'static str, fn(WeightDtype) -> Box<dyn Trainer>);

const FAMILIES: [Family; 2] = [("Mamba-2", m2), ("MIMO", mimo)];

const DTYPES: [WeightDtype; 4] = [
    WeightDtype::F32,
    WeightDtype::Tf32,
    WeightDtype::Bf16,
    WeightDtype::F16,
];

/// What a run left: the master bits, every step's metrics, the scale.
type Run = (
    Vec<u32>,
    Vec<(u64, bool, Option<f32>, Option<bool>)>,
    Option<(f32, u32)>,
);

/// Five steps, the learning rate halved before the fourth, the third
/// overflowing in f16; `graph` captures after the first.
fn run(make: fn(WeightDtype) -> Box<dyn Trainer>, dtype: WeightDtype, graph: bool) -> Run {
    let mut t = make(dtype);
    let mut metrics = Vec::new();
    for k in 0..5u32 {
        if k == 3 {
            t.set_lr(5e-4);
        }
        let (input, d) = window(k, k == 2 && dtype == WeightDtype::F16);
        let m = t.step(&input, &d).unwrap();
        assert_eq!(m.graph_replayed, graph && k > 0, "{dtype:?} step {k}");
        metrics.push((m.step, m.graph_replayed, m.loss_scale, m.overflow_skipped));
        if graph && k == 0 {
            t.capture().unwrap();
        }
    }
    let replayed: Vec<_> = metrics.iter().map(|m| (m.0, false, m.2, m.3)).collect();
    (t.master(), replayed, t.scaler())
}

#[test]
fn the_captured_step_replays_the_eager_bits() {
    for (family, make) in FAMILIES {
        for dtype in DTYPES {
            let eager = run(make, dtype, false);
            let graph = run(make, dtype, true);
            assert_eq!(graph.1, eager.1, "{family} {dtype:?}: the steps' metrics");
            assert_eq!(graph.2, eager.2, "{family} {dtype:?}: the loss scale");
            assert_eq!(graph.0, eager.0, "{family} {dtype:?}: the master weights");
            if dtype == WeightDtype::F16 {
                assert_eq!(eager.1[2].3, Some(true), "{family}: the overflow skipped");
                assert_eq!(eager.1[2].0, eager.1[1].0, "{family}: no step counted");
            }
        }
    }
}

#[test]
fn a_baked_adamw_setting_needs_a_new_capture() {
    for (family, make) in FAMILIES {
        let mut t = make(WeightDtype::Bf16);
        let (input, d) = window(0, false);
        t.step(&input, &d).unwrap();
        t.capture().unwrap();
        assert!(t.step(&input, &d).unwrap().graph_replayed, "{family}");
        t.set_beta1(0.8);
        assert!(t.step(&input, &d).is_err(), "{family}: a changed beta1");
        t.drop_graph();
        let m = t.step(&input, &d).unwrap();
        assert!(!m.graph_replayed, "{family}: eager after the drop");
        t.capture().unwrap();
        assert!(
            t.step(&input, &d).unwrap().graph_replayed,
            "{family}: captured again"
        );
    }
}

/// An open accumulation window folds into the next step, which the graph
/// does not do, so that step runs eagerly, with the eager bits.
#[test]
fn an_open_window_runs_the_step_eagerly() {
    for (family, make) in FAMILIES {
        let mut runs = Vec::new();
        for graph in [false, true] {
            let mut t = make(WeightDtype::F32);
            let (input, d) = window(0, false);
            t.step(&input, &d).unwrap();
            if graph {
                t.capture().unwrap();
            }
            let (input, d) = window(1, false);
            t.accumulate(&input, &d);
            let m = t.step(&input, &d).unwrap();
            assert!(!m.graph_replayed, "{family}: the window's step");
            let (input, d) = window(2, false);
            assert_eq!(
                t.step(&input, &d).unwrap().graph_replayed,
                graph,
                "{family}"
            );
            runs.push(t.master());
        }
        assert_eq!(runs[0], runs[1], "{family}");
    }
}
