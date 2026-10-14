//! A bidirectional pair is its two layers composed by hand, on the host.
//!
//! The pair reads its input `x` twice: its first layer forward, its second
//! with every sequence reversed, and the two mixers' outputs join the
//! residual stream together. Built from two one-layer segments of the same
//! weights, `F` (the first layer) and `G` (the second), that is
//! `F(x) + R(G(R(x))) - x`, with `R` the reversal of each sequence: the
//! pair's rows are that, its input gradient is `F'(g) + R(G'(R(g))) - g`,
//! and its weight gradients are F's and G's to the bit, the second layer's
//! taken on the reversed window. With and without a packing, which turns
//! every sequence within its own span.

use mamba_rs::{
    BackwardOpts, CpuTrainer, Mamba2Config, Mamba2CpuTrainer, Mamba2Weights, Mamba3Config,
    Mamba3CpuTrainer, Mamba3MimoConfig, Mamba3MimoCpuTrainer, Mamba3MimoWeights, Mamba3Weights,
    MambaConfig, MambaCpuTrainer, MambaWeights, TrainSessionCfg, Varlen,
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

fn run<T: CpuTrainer>(t: &mut T, packing: Option<&Varlen>, (x, d): (&[f32], &[f32])) -> Run {
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
fn check_pair<T: CpuTrainer>(
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
fn a_mamba2_pair_is_its_layers_composed() {
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
    for p in PACKINGS {
        let trainers = [
            Mamba2CpuTrainer::new(&w, m2_cfg(2, true), s).unwrap(),
            Mamba2CpuTrainer::new(&layer(0), m2_cfg(1, false), s).unwrap(),
            Mamba2CpuTrainer::new(&layer(1), m2_cfg(1, false), s).unwrap(),
        ];
        check_pair("Mamba-2", trainers, &starts_of(p));
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
fn a_mamba3_siso_pair_is_its_layers_composed() {
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
    for p in PACKINGS {
        let trainers = [
            Mamba3CpuTrainer::new(&w, siso_cfg(2, true), s).unwrap(),
            Mamba3CpuTrainer::new(&layer(0), siso_cfg(1, false), s).unwrap(),
            Mamba3CpuTrainer::new(&layer(1), siso_cfg(1, false), s).unwrap(),
        ];
        check_pair("Mamba-3 SISO", trainers, &starts_of(p));
    }
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
fn a_mamba1_pair_is_its_layers_composed() {
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
    let trainers = [
        MambaCpuTrainer::new(&w, m1_cfg(2, true), s).unwrap(),
        MambaCpuTrainer::new(&layer(0), m1_cfg(1, false), s).unwrap(),
        MambaCpuTrainer::new(&layer(1), m1_cfg(1, false), s).unwrap(),
    ];
    check_pair("Mamba-1", trainers, &starts_of(PACKINGS[0]));
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
fn a_mamba3_mimo_pair_is_its_layers_composed() {
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
    for p in PACKINGS {
        let trainers = [
            Mamba3MimoCpuTrainer::new(&w, mimo_cfg(2, true), s).unwrap(),
            Mamba3MimoCpuTrainer::new(&layer(0), mimo_cfg(1, false), s).unwrap(),
            Mamba3MimoCpuTrainer::new(&layer(1), mimo_cfg(1, false), s).unwrap(),
        ];
        check_pair("Mamba-3 MIMO", trainers, &starts_of(p));
    }
}
