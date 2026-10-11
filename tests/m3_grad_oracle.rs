//! Mamba-3 SISO training gradients on the CPU against central differences
//! of the float64 reference step.
//!
//! The f32 finite differences in m3_test_correctness.rs need coarse steps
//! and a 5% tolerance, and they cannot touch in_proj at all; differencing
//! the float64 step drops the floor far enough to hold every tensor, every
//! in_proj column group and the layer inputs to 5e-3, from a fresh state
//! and from one carried in from earlier tokens.

use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::backward::backward_mamba3_layer_batched;
use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
use mamba_rs::mamba3_siso::cpu::flat::Mamba3LayerFlat;
use mamba_rs::mamba3_siso::cpu::forward::{Mamba3LayerStateMut, forward_mamba3_layer_batched};
use mamba_rs::mamba3_siso::cpu::scratch::Mamba3Scratch;
use mamba_rs::mamba3_siso::cpu::weights::TrainMamba3LayerWeights;

#[path = "common/m3_oracle.rs"]
mod m3_oracle;
use m3_oracle::{Layer64, layer_step64, zeros64};

fn det(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// One layer's weights with every parameter drawn.
fn drawn(dims: &Mamba3Dims, seed: u64) -> TrainMamba3LayerWeights {
    let mut w = TrainMamba3LayerWeights::zeros(dims);
    for (v, k, base, scale) in [
        (&mut w.norm_weight, 1, 1.0, 0.2),
        (&mut w.in_proj_w, 2, 0.0, 0.3),
        (&mut w.dt_bias, 3, -1.0, 0.5),
        (&mut w.b_norm_weight, 4, 1.0, 0.3),
        (&mut w.c_norm_weight, 5, 1.0, 0.3),
        (&mut w.b_bias, 6, 0.0, 0.5),
        (&mut w.c_bias, 7, 0.0, 0.5),
        (&mut w.d_param, 8, 1.0, 0.5),
        (&mut w.norm_gate_weight, 9, 1.0, 0.3),
        (&mut w.out_proj_w, 10, 0.0, 0.3),
    ] {
        let draws = det(v.len(), seed * 100 + k, scale);
        for (x, dv) in v.iter_mut().zip(draws) {
            *x = base + dv;
        }
    }
    w
}

/// The f32 states of a stack, one set per layer.
#[derive(Clone)]
struct States {
    ssm: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    angle: Vec<Vec<f32>>,
}

impl States {
    fn zeros(dims: &Mamba3Dims, layers: usize) -> Self {
        let (nh, hd, ds, na) = (
            dims.nheads,
            dims.headdim,
            dims.d_state,
            dims.num_rope_angles.max(1),
        );
        Self {
            ssm: vec![vec![0.0; nh * hd * ds]; layers],
            k: vec![vec![0.0; nh * ds]; layers],
            v: vec![vec![0.0; nh * hd]; layers],
            angle: vec![vec![0.0; nh * na]; layers],
        }
    }

    /// The f64 copy for the oracle; the layouts are the same.
    fn to64(&self, na: usize) -> Vec<Layer64> {
        let f = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
        (0..self.ssm.len())
            .map(|l| Layer64 {
                ssm: f(&self.ssm[l]),
                k: f(&self.k[l]),
                v: f(&self.v[l]),
                angle: if na == 0 {
                    Vec::new()
                } else {
                    f(&self.angle[l])
                },
            })
            .collect()
    }
}

/// The stack's training forward over a window: the residual stream in
/// `temporal`, every layer's saved activations, the states carried.
fn forward(
    weights: &[TrainMamba3LayerWeights],
    dims: &Mamba3Dims,
    temporal: &mut [f32],
    states: &mut States,
) -> Vec<Mamba3LayerFlat> {
    let mut scratch = Mamba3Scratch::zeros(dims);
    weights
        .iter()
        .enumerate()
        .map(|(l, lw)| {
            let mut acts = Mamba3LayerFlat::zeros(*dims);
            forward_mamba3_layer_batched(
                temporal,
                &mut acts,
                lw,
                Mamba3LayerStateMut {
                    ssm: &mut states.ssm[l],
                    k: &mut states.k[l],
                    v: &mut states.v[l],
                    angle: &mut states.angle[l],
                },
                &mut scratch,
                dims,
            );
            acts
        })
        .collect()
}

/// The probed loss `sum_t <out_t, probe_t>` of the stack in float64, from
/// `start`, which is an input of the window like the rows are.
fn loss64(
    cfg: &Mamba3Config,
    weights: &[TrainMamba3LayerWeights],
    (inputs, probe): (&[f32], &[f32]),
    start: &[Layer64],
) -> f64 {
    let dm = cfg.d_model;
    let mut st = start.to_vec();
    inputs
        .chunks_exact(dm)
        .zip(probe.chunks_exact(dm))
        .map(|(row, pr)| {
            let mut x: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            for (lw, s) in weights.iter().zip(st.iter_mut()) {
                x = layer_step64(cfg, lw, s, &x);
            }
            x.iter()
                .zip(pr)
                .map(|(o, &p)| o * f64::from(p))
                .sum::<f64>()
        })
        .sum()
}

fn tensors(w: &mut TrainMamba3LayerWeights) -> [(&'static str, &mut Vec<f32>); 10] {
    [
        ("norm", &mut w.norm_weight),
        ("in_proj_w", &mut w.in_proj_w),
        ("dt_bias", &mut w.dt_bias),
        ("b_norm", &mut w.b_norm_weight),
        ("c_norm", &mut w.c_norm_weight),
        ("b_bias", &mut w.b_bias),
        ("c_bias", &mut w.c_bias),
        ("d", &mut w.d_param),
        ("norm_gate", &mut w.norm_gate_weight),
        ("out_proj_w", &mut w.out_proj_w),
    ]
}

fn cases() -> Vec<(&'static str, Mamba3Config)> {
    let base = Mamba3Config {
        d_model: 16,
        d_state: 8,
        headdim: 8,
        expand: 2,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        is_outproj_norm: false,
        ..Mamba3Config::default()
    };
    vec![
        ("groups 1", base),
        (
            "groups 2, full rope",
            Mamba3Config {
                ngroups: 2,
                rope_fraction: 1.0,
                ..base
            },
        ),
        (
            "output norm",
            Mamba3Config {
                is_outproj_norm: true,
                ..base
            },
        ),
        (
            "clamped A",
            Mamba3Config {
                a_floor: 0.5,
                ..base
            },
        ),
    ]
}

/// The probed entries of a gradient: its largest, a spread that does not
/// alias onto one column, and for in_proj one entry of every column group.
fn probe_indices(cfg: &Mamba3Config, name: &str, g: &[f32]) -> Vec<usize> {
    let len = g.len();
    let argmax = g
        .iter()
        .enumerate()
        .fold((0, 0f32), |(bi, bv), (i, &v)| {
            if v.abs() > bv { (i, v.abs()) } else { (bi, bv) }
        })
        .0;
    let mut idx = vec![argmax];
    idx.extend((1..5).map(|k| (k * (len / 5) + 7 * k + 3) % len));
    if name == "in_proj_w" {
        let (di, gn, nh, na) = (
            cfg.d_inner(),
            cfg.ngroups * cfg.d_state,
            cfg.nheads(),
            cfg.num_rope_angles(),
        );
        let out = cfg.in_proj_out_dim();
        let rows = len / out;
        let heads = 2 * di + 2 * gn;
        let groups = [
            1,
            di + 1,
            2 * di + 1,
            2 * di + gn + 1,
            heads,
            heads + nh,
            heads + 2 * nh,
            heads + 3 * nh,
        ];
        for (k, &col) in groups.iter().enumerate() {
            if col < out && (col < heads + 3 * nh || na > 0) {
                idx.push(((3 * k + 1) % rows) * out + col);
            }
        }
    }
    idx
}

/// The analytic f32 gradient of every tensor of every layer (at its largest
/// entry, a spread and every in_proj column group) and of the inputs
/// against float64 central differences, for windows that start from zero
/// and from a state carried in from earlier tokens.
#[test]
fn m3_gradients_match_f64_central_differences() {
    let (seq, warm) = (9usize, 5usize);
    for (label, cfg) in cases() {
        cfg.validate().unwrap();
        let dims = Mamba3Dims::from_config(&cfg, seq);
        let warm_dims = Mamba3Dims::from_config(&cfg, warm);
        let dm = cfg.d_model;
        let na = cfg.num_rope_angles();
        let weights: Vec<TrainMamba3LayerWeights> = (0..cfg.n_layers)
            .map(|l| drawn(&dims, 31 + l as u64))
            .collect();
        let inputs = det(seq * dm, 77, 1.0);
        let probe = det(seq * dm, 78, 1.0);
        let mut carried = States::zeros(&dims, cfg.n_layers);
        forward(
            &weights,
            &warm_dims,
            &mut det(warm * dm, 79, 1.0),
            &mut carried,
        );
        for (start_label, start) in [
            ("zero", States::zeros(&dims, cfg.n_layers)),
            ("carried", carried.clone()),
        ] {
            let start64 = if start_label == "zero" {
                zeros64(&cfg)
            } else {
                start.to64(na)
            };
            let angle_init = start.angle.clone();
            let mut st = start.clone();
            let mut temporal = inputs.clone();
            let acts = forward(&weights, &dims, &mut temporal, &mut st);
            let mut grads: Vec<TrainMamba3LayerWeights> = (0..cfg.n_layers)
                .map(|_| TrainMamba3LayerWeights::zeros(&dims))
                .collect();
            let mut d_temporal = probe.clone();
            let mut scratch = Mamba3Scratch::zeros(&dims);
            for l in (0..cfg.n_layers).rev() {
                backward_mamba3_layer_batched(
                    &mut d_temporal,
                    &acts[l],
                    &weights[l],
                    &mut grads[l],
                    &mut scratch,
                    &dims,
                    Some(&angle_init[l]),
                );
            }
            let loss = |ws: &[TrainMamba3LayerWeights], xs: &[f32]| {
                loss64(&cfg, ws, (xs, &probe), &start64)
            };
            let mut checked = 0usize;
            let mut worst = 0f64;
            for l in 0..cfg.n_layers {
                let mut g_layer = grads[l].clone();
                let names_and_grads: Vec<(&str, Vec<f32>)> = tensors(&mut g_layer)
                    .into_iter()
                    .map(|(n, v)| (n, v.clone()))
                    .collect();
                for (name, g) in &names_and_grads {
                    if *name == "norm_gate" && !cfg.is_outproj_norm {
                        continue;
                    }
                    let scale = g.iter().fold(0f32, |a, &v| a.max(v.abs()));
                    assert!(scale > 0.0, "{label}: layer {l} {name} has no gradient");
                    for i in probe_indices(&cfg, name, g) {
                        let perturb = |sign: f32| -> (Vec<TrainMamba3LayerWeights>, f64) {
                            let mut wp = weights.clone();
                            let mut tp = tensors(&mut wp[l]);
                            let v = &mut tp.iter_mut().find(|(n, _)| n == name).unwrap().1;
                            v[i] += sign * 1e-3 * v[i].abs().max(1.0);
                            let at = f64::from(v[i]);
                            drop(tp);
                            (wp, at)
                        };
                        let (plus, hp) = perturb(1.0);
                        let (minus, hm) = perturb(-1.0);
                        let fd = (loss(&plus, &inputs) - loss(&minus, &inputs)) / (hp - hm);
                        let err =
                            (f64::from(g[i]) - fd).abs() / fd.abs().max(0.05 * f64::from(scale));
                        worst = worst.max(err);
                        assert!(
                            err < 5e-3,
                            "{label}, {start_label} start: layer {l} {name}[{i}] analytic {} \
                             vs fd {fd:.6e} (rel {err:.3e})",
                            g[i]
                        );
                        checked += 1;
                    }
                }
            }
            for i in (0..d_temporal.len()).step_by(7) {
                let mut xp = inputs.clone();
                let mut xm = inputs.clone();
                xp[i] += 1e-3;
                xm[i] -= 1e-3;
                let fd = (loss(&weights, &xp) - loss(&weights, &xm)) / f64::from(xp[i] - xm[i]);
                let err = (f64::from(d_temporal[i]) - fd).abs() / fd.abs().max(1e-2);
                assert!(
                    err < 5e-3,
                    "{label}, {start_label} start: d_input[{i}] {} vs fd {fd:.6e}",
                    d_temporal[i]
                );
            }
            eprintln!(
                "{label}, {start_label} start: {checked} weight probes, worst rel {worst:.2e}"
            );
        }
    }
}
