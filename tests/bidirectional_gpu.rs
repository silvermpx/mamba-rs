//! A bidirectional pair on the device is its two layers composed by hand,
//! as on the host (see `bidirectional_cpu.rs`): its rows and input
//! gradient within float tolerance of `F(x) + R(G(R(x))) - x` and its
//! weight gradients F's and G's to the bit, the second layer's taken on the
//! reversed window; and the device pair agrees with the host pair.
#![cfg(feature = "cuda")]

use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::{
    BackwardOpts, CpuTrainer, GpuTrainer, Mamba2Config, Mamba2CpuTrainer, Mamba2GpuTrainer,
    Mamba2Weights, Mamba3Config, Mamba3CpuTrainer, Mamba3GpuTrainer, Mamba3MimoConfig,
    Mamba3MimoCpuTrainer, Mamba3MimoGpuTrainer, Mamba3MimoWeights, Mamba3Weights, MambaConfig,
    MambaCpuTrainer, MambaGpuTrainer, MambaWeights, TrainSessionCfg, Varlen,
};

const D_MODEL: usize = 16;
const BATCH: usize = 2;
const SEQ_LEN: usize = 13;

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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// ||a - b|| / ||a|| in f64.
fn rel(a: &[f64], b: &[f32]) -> f64 {
    let (num, den) = a.iter().zip(b).fold((0.0, 0.0), |(n, d), (&x, &y)| {
        let e = x - f64::from(y);
        (n + e * e, d + x * x)
    });
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

/// Every sequence of every batch row reversed: `starts[b]` are the
/// positions where row b's sequences begin after the first.
fn reversed(rows: &[f32], starts: &[Vec<usize>]) -> Vec<f32> {
    let mut out = rows.to_vec();
    for (b, row_starts) in starts.iter().enumerate() {
        let row = &mut out[b * SEQ_LEN * D_MODEL..(b + 1) * SEQ_LEN * D_MODEL];
        let mut bounds: Vec<usize> = std::iter::once(0)
            .chain(row_starts.iter().copied().filter(|&t| t > 0))
            .collect();
        bounds.push(SEQ_LEN);
        for w in bounds.windows(2) {
            let (first, end) = (w[0], w[1]);
            for i in 0..(end - first) / 2 {
                let (a, c) = (first + i, end - 1 - i);
                for k in 0..D_MODEL {
                    row.swap(a * D_MODEL + k, c * D_MODEL + k);
                }
            }
        }
    }
    out
}

/// `a + R(b) - x`, in f64.
fn composed(a: &[f32], b_rev: &[f32], x: &[f32]) -> Vec<f64> {
    a.iter()
        .zip(b_rev)
        .zip(x)
        .map(|((&a, &b), &x)| f64::from(a) + f64::from(b) - f64::from(x))
        .collect()
}

/// The run of one trainer on a window: its rows, its input gradient for
/// the output gradient `d` and its gradient arena.
struct Run {
    rows: Vec<f32>,
    d_input: Vec<f32>,
    grads: Vec<f32>,
}

fn run<T: GpuTrainer>(t: &mut T, packing: Option<&Varlen>, (x, d): (&[f32], &[f32])) -> Run {
    t.set_varlen(packing).unwrap();
    let mut rows = vec![0.0f32; x.len()];
    t.forward(x, &mut rows).unwrap();
    let mut d_input = vec![0.0f32; x.len()];
    t.backward_step_input_grad(
        d,
        BackwardOpts::default().with_accumulate_only(true),
        &mut d_input,
    )
    .unwrap();
    Run {
        rows,
        d_input,
        grads: t.download_grad_arena().unwrap(),
    }
}

/// The pair against its composition, for one packing (`starts` per batch
/// row, empty for one sequence a row).
fn check_pair<T: GpuTrainer>(
    family: &str,
    [mut pair, mut first, mut second]: [T; 3],
    starts: &[Vec<usize>],
) {
    let packing =
        (!starts.iter().all(Vec::is_empty)).then(|| Varlen::from_starts(starts, SEQ_LEN).unwrap());
    let x = values(BATCH * SEQ_LEN * D_MODEL, 11);
    let d: Vec<f32> = values(BATCH * SEQ_LEN * D_MODEL, 12)
        .iter()
        .map(|v| v * 0.01)
        .collect();
    let case = format!("{family}, sequences starting at {starts:?}");
    let p = run(&mut pair, packing.as_ref(), (&x, &d));
    let f = run(&mut first, packing.as_ref(), (&x, &d));
    let (xr, dr) = (reversed(&x, starts), reversed(&d, starts));
    let g = run(&mut second, packing.as_ref(), (&xr, &dr));

    let want = composed(&f.rows, &reversed(&g.rows, starts), &x);
    let err = rel(&want, &p.rows);
    assert!(err < 1e-6, "{case}: the pair's rows are off by {err:e}");
    let want = composed(&f.d_input, &reversed(&g.d_input, starts), &d);
    let err = rel(&want, &p.d_input);
    assert!(
        err < 1e-5,
        "{case}: the pair's input gradient is off by {err:e}"
    );
    let mut halves = f.grads;
    halves.extend_from_slice(&g.grads);
    assert_eq!(
        bits(&p.grads),
        bits(&halves),
        "{case}: the pair's weight gradients"
    );
}

const PACKINGS: [[&[usize]; BATCH]; 3] = [[&[], &[]], [&[0, 5], &[0, 4, 9]], [&[3], &[7, 8]]];

fn starts_of(p: [&[usize]; BATCH]) -> Vec<Vec<usize>> {
    p.iter().map(|s| s.to_vec()).collect()
}

/// The lanes every family's pair runs on: f32 and the bf16 mixed lane.
const LANES: [WeightDtype; 2] = [WeightDtype::F32, WeightDtype::Bf16];

fn opts(dtype: WeightDtype) -> GpuOptions {
    GpuOptions::default().dtype(dtype)
}

fn m2_cfg(n_layers: usize, bidirectional: bool) -> Mamba2Config {
    Mamba2Config {
        d_model: D_MODEL,
        d_state: 8,
        headdim: 8,
        n_layers,
        final_norm: false,
        bidirectional,
        ..Mamba2Config::default()
    }
}

#[test]
fn a_mamba2_pair_on_the_device_is_its_layers_composed() {
    let s = TrainSessionCfg::new(D_MODEL, BATCH, SEQ_LEN);
    let mut w = Mamba2Weights::init(&m2_cfg(2, true), D_MODEL, 5);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    w.norm_f_weight.clear();
    w.norm_f_bias.clear();
    let layer = |l: usize| {
        let mut one = w.clone();
        one.layers = vec![w.layers[l].clone()];
        one
    };
    for dtype in LANES {
        let gpu = |w: &Mamba2Weights, cfg| Mamba2GpuTrainer::new(w, cfg, s, opts(dtype)).unwrap();
        for p in PACKINGS {
            let starts = starts_of(p);
            let trainers = [
                gpu(&w, m2_cfg(2, true)),
                gpu(&layer(0), m2_cfg(1, false)),
                gpu(&layer(1), m2_cfg(1, false)),
            ];
            check_pair(&format!("Mamba-2 {dtype:?}"), trainers, &starts);
            if dtype == WeightDtype::F32 {
                let cfg = Mamba2Config {
                    d_intermediate: 24,
                    ..m2_cfg(2, true)
                };
                let wm = Mamba2Weights::init(&cfg, D_MODEL, 9);
                let mut host = Mamba2CpuTrainer::new(&wm, cfg, s).unwrap();
                let mut dev = gpu(&wm, cfg);
                device_against_host("Mamba-2 with MLPs", (&mut dev, &mut host), &starts);
            }
        }
    }
}

fn siso_cfg(n_layers: usize, bidirectional: bool) -> Mamba3Config {
    Mamba3Config {
        d_model: D_MODEL,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers,
        final_norm: false,
        bidirectional,
        ..Mamba3Config::default()
    }
}

#[test]
fn a_mamba3_siso_pair_on_the_device_is_its_layers_composed() {
    let s = TrainSessionCfg::new(D_MODEL, BATCH, SEQ_LEN);
    let mut w = Mamba3Weights::init(&siso_cfg(2, true), D_MODEL, 6);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    w.norm_f_weight.clear();
    w.norm_f_bias.clear();
    let layer = |l: usize| {
        let mut one = w.clone();
        one.layers = vec![w.layers[l].clone()];
        one
    };
    for dtype in LANES {
        let gpu = |w: &Mamba3Weights, cfg| Mamba3GpuTrainer::new(w, cfg, s, opts(dtype)).unwrap();
        for p in PACKINGS {
            let starts = starts_of(p);
            let trainers = [
                gpu(&w, siso_cfg(2, true)),
                gpu(&layer(0), siso_cfg(1, false)),
                gpu(&layer(1), siso_cfg(1, false)),
            ];
            check_pair(&format!("Mamba-3 SISO {dtype:?}"), trainers, &starts);
            if dtype == WeightDtype::F32 {
                let cfg = Mamba3Config {
                    d_intermediate: 24,
                    ..siso_cfg(2, true)
                };
                let wm = Mamba3Weights::init(&cfg, D_MODEL, 9);
                let mut host = Mamba3CpuTrainer::new(&wm, cfg, s).unwrap();
                let mut dev = gpu(&wm, cfg);
                device_against_host("Mamba-3 SISO with MLPs", (&mut dev, &mut host), &starts);
            }
        }
    }
}

fn mimo_cfg(n_layers: usize, bidirectional: bool) -> Mamba3MimoConfig {
    Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        mimo_rank: 2,
        n_layers,
        final_norm: false,
        bidirectional,
        ..Mamba3MimoConfig::default()
    }
}

#[test]
fn a_mamba3_mimo_pair_on_the_device_is_its_layers_composed() {
    let s = TrainSessionCfg::new(D_MODEL, BATCH, SEQ_LEN);
    let mut w = Mamba3MimoWeights::init(&mimo_cfg(2, true), D_MODEL, 8);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    w.norm_f_weight.clear();
    w.norm_f_bias.clear();
    let layer = |l: usize| {
        let mut one = w.clone();
        one.layers = vec![w.layers[l].clone()];
        one
    };
    for dtype in LANES {
        let gpu =
            |w: &Mamba3MimoWeights, cfg| Mamba3MimoGpuTrainer::new(w, cfg, s, opts(dtype)).unwrap();
        for p in PACKINGS {
            let starts = starts_of(p);
            let trainers = [
                gpu(&w, mimo_cfg(2, true)),
                gpu(&layer(0), mimo_cfg(1, false)),
                gpu(&layer(1), mimo_cfg(1, false)),
            ];
            check_pair(&format!("Mamba-3 MIMO {dtype:?}"), trainers, &starts);
            if dtype == WeightDtype::F32 {
                let cfg = Mamba3MimoConfig {
                    d_intermediate: 24,
                    ..mimo_cfg(2, true)
                };
                let wm = Mamba3MimoWeights::init(&cfg, D_MODEL, 9);
                let mut host = Mamba3MimoCpuTrainer::new(&wm, cfg, s).unwrap();
                let mut dev = gpu(&wm, cfg);
                device_against_host("Mamba-3 MIMO with MLPs", (&mut dev, &mut host), &starts);
            }
        }
    }
}

/// The device pair's rows and input gradient against the host pair's for
/// one packing.
fn device_against_host<G: GpuTrainer, C: CpuTrainer>(
    family: &str,
    (dev, host): (&mut G, &mut C),
    starts: &[Vec<usize>],
) {
    let packing =
        (!starts.iter().all(Vec::is_empty)).then(|| Varlen::from_starts(starts, SEQ_LEN).unwrap());
    let x = values(BATCH * SEQ_LEN * D_MODEL, 11);
    let d: Vec<f32> = values(BATCH * SEQ_LEN * D_MODEL, 12)
        .iter()
        .map(|v| v * 0.01)
        .collect();
    host.set_varlen(packing.as_ref()).unwrap();
    let mut want_rows = vec![0.0f32; x.len()];
    host.forward(&x, &mut want_rows).unwrap();
    let mut want_dx = vec![0.0f32; x.len()];
    host.backward_step_input_grad(
        &d,
        BackwardOpts::default().with_accumulate_only(true),
        &mut want_dx,
    )
    .unwrap();
    let got = run(dev, packing.as_ref(), (&x, &d));
    let f64s = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let err = rel(&f64s(&want_rows), &got.rows);
    assert!(
        err < 1e-5,
        "{family} {starts:?}: the device pair's rows are off the host pair's by {err:e}"
    );
    let err = rel(&f64s(&want_dx), &got.d_input);
    assert!(
        err < 1e-4,
        "{family} {starts:?}: the device pair's input gradient is off the host's by {err:e}"
    );
}

fn m1_cfg(n_layers: usize, bidirectional: bool) -> MambaConfig {
    MambaConfig {
        d_model: D_MODEL,
        d_state: 8,
        n_layers,
        final_norm: false,
        bidirectional,
        ..MambaConfig::default()
    }
}

/// Mamba-1 packs no sequences, so its pairs reverse whole batch rows.
#[test]
fn a_mamba1_pair_on_the_device_is_its_layers_composed() {
    let s = TrainSessionCfg::new(D_MODEL, BATCH, SEQ_LEN);
    let mut w = MambaWeights::init(&m1_cfg(2, true), D_MODEL, 7);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    w.norm_f_weight.clear();
    w.norm_f_bias.clear();
    let layer = |l: usize| {
        let mut one = w.clone();
        one.layers = vec![w.layers[l].clone()];
        one
    };
    let starts = starts_of(PACKINGS[0]);
    for dtype in LANES {
        let gpu = |w: &MambaWeights, cfg| MambaGpuTrainer::new(w, cfg, s, opts(dtype)).unwrap();
        let trainers = [
            gpu(&w, m1_cfg(2, true)),
            gpu(&layer(0), m1_cfg(1, false)),
            gpu(&layer(1), m1_cfg(1, false)),
        ];
        check_pair(&format!("Mamba-1 {dtype:?}"), trainers, &starts);
        if dtype == WeightDtype::F32 {
            let cfg = MambaConfig {
                d_intermediate: 24,
                ..m1_cfg(2, true)
            };
            let wm = MambaWeights::init(&cfg, D_MODEL, 9);
            let mut host = MambaCpuTrainer::new(&wm, cfg, s).unwrap();
            let mut dev = gpu(&wm, cfg);
            device_against_host("Mamba-1 with MLPs", (&mut dev, &mut host), &starts);
        }
    }
}

#[test]
fn a_bidirectional_mamba2_engine_refuses_the_decode_step() {
    use mamba_rs::gpu::device::GpuDevice;
    use mamba_rs::mamba2::gpu::Mamba2GpuEngine;
    let cfg = m2_cfg(2, true);
    let w = Mamba2Weights::init(&cfg, D_MODEL, 5);
    let device = GpuDevice::new(0).unwrap();
    let engine = Mamba2GpuEngine::new(&device, &w, cfg, D_MODEL, BATCH, WeightDtype::F32).unwrap();
    let mut state = engine.alloc_state().unwrap();
    let mut scratch = engine.alloc_scratch().unwrap();
    let x = values(BATCH * D_MODEL, 3);
    let err = engine
        .step_gpu_only(&x, &mut state, &mut scratch)
        .unwrap_err();
    assert!(err.contains("no decode step"), "{err}");
}
