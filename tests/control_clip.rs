//! The control-channel clip bounds the gradients of the step size and the
//! decay on their own, before the global clip. After a step asked for it,
//! the arena holds the window's gradients with exactly those channels
//! scaled down to the bound and every other element untouched, and the
//! reported norm is that arena's. The test finds the channels from each
//! family's gradient layout by itself, so a region that drifts from the
//! layout fails here.
#![cfg(feature = "cuda")]

use mamba_rs::{
    BackwardOpts, GpuOptions, Mamba2Config, Mamba2GpuTrainer, Mamba2Weights, Mamba3Config,
    Mamba3GpuTrainer, Mamba3MimoConfig, Mamba3MimoGpuTrainer, Mamba3MimoWeights, Mamba3Weights,
    MambaConfig, MambaGpuTrainer, MambaWeights, TrainSessionCfg, WeightDtype,
};

const BATCH: usize = 2;
const SEQ_LEN: usize = 48;

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.5
        })
        .collect()
}

fn session(d_model: usize) -> TrainSessionCfg {
    TrainSessionCfg {
        input_dim: d_model,
        batch: BATCH,
        seq_len: SEQ_LEN,
        lr: 1e-3,
        weight_decay: 0.1,
    }
}

/// Mark `rows` rows of `cols` columns starting at `off`, each `stride`
/// apart.
fn mark_stripe(mask: &mut [bool], off: usize, rows: usize, cols: usize, stride: usize) {
    for r in 0..rows {
        mask[off + r * stride..off + r * stride + cols].fill(true);
    }
}

/// The control channels of a Mamba-3 SISO arena: the last 3 * nheads +
/// angles columns of every layer's in_proj gradient, then dt_bias.
fn siso_control(cfg: &Mamba3Config, len: usize) -> Vec<bool> {
    let (dm, di, ds, nh) = (cfg.d_model, cfg.d_inner(), cfg.d_state, cfg.nheads());
    let ip = cfg.in_proj_out_dim();
    let ctrl = 3 * nh + cfg.num_rope_angles();
    let per_layer = dm + dm * ip + nh + 2 * ds + 2 * nh * ds + nh + di + di * dm;
    let mut mask = vec![false; len];
    for l in 0..cfg.n_layers {
        let in_proj = dm * dm + dm + l * per_layer + dm;
        mark_stripe(&mut mask, in_proj + ip - ctrl, dm, ctrl, ip);
        let dt_bias = in_proj + dm * ip;
        mask[dt_bias..dt_bias + nh].fill(true);
    }
    mask
}

/// The control channels of a Mamba-1 arena: the step-size columns of
/// every layer's x_proj gradient, then dt_proj's weight and bias and
/// A_log.
fn m1_control(cfg: &MambaConfig, len: usize) -> Vec<bool> {
    let (dm, di, ds, dc) = (cfg.d_model, cfg.d_inner(), cfg.d_state, cfg.d_conv);
    let (dr, xd) = (cfg.dt_rank(), cfg.xdbl_dim());
    let per_layer =
        dm + dm * 2 * di + di * dc + di + di * xd + dr * di + di + di * ds + di + di * dm;
    let mut mask = vec![false; len];
    for l in 0..cfg.n_layers {
        let x_proj = dm * dm + dm + l * per_layer + dm + dm * 2 * di + di * dc + di;
        mark_stripe(&mut mask, x_proj, di, dr, xd);
        let dt_proj = x_proj + di * xd;
        mask[dt_proj..dt_proj + dr * di + di + di * ds].fill(true);
    }
    mask
}

/// A Mamba-2 arena: the input projection, every layer's mixer tensors in
/// field order, every layer's MLP, the final norm.
fn m2_arena(w: &Mamba2Weights) -> Vec<f32> {
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
    for m in &w.mlp {
        for t in [&m.norm_weight, &m.fc1_w, &m.fc2_w] {
            out.extend_from_slice(t);
        }
    }
    out.extend_from_slice(&w.norm_f_weight);
    out
}

/// The control channels of a Mamba-2 arena: the dt columns of every
/// layer's in_proj gradient (the last nheads of its split), their in_proj
/// bias entries where the model has the bias, then dt_bias and A_log.
fn m2_control(cfg: &Mamba2Config, w: &Mamba2Weights) -> Vec<bool> {
    let (dm, ip, nh) = (cfg.d_model, cfg.in_proj_out_dim(), cfg.nheads());
    let mut mask = vec![false; m2_arena(w).len()];
    let mut at = w.input_proj_w.len() + w.input_proj_b.len();
    for l in &w.layers {
        let in_proj = at + l.norm_weight.len();
        mark_stripe(&mut mask, in_proj + ip - nh, dm, nh, ip);
        let bias = in_proj + l.in_proj_w.len();
        if !l.in_proj_b.is_empty() {
            mask[bias + ip - nh..bias + ip].fill(true);
        }
        let dt_bias = bias + l.in_proj_b.len() + l.conv_w.len() + l.conv_b.len();
        mask[dt_bias..dt_bias + l.dt_bias.len() + l.a_log.len()].fill(true);
        at = dt_bias
            + l.dt_bias.len()
            + l.a_log.len()
            + l.d_param.len()
            + l.gate_norm_weight.len()
            + l.out_proj_w.len()
            + l.out_proj_b.len();
    }
    mask
}

/// A MIMO arena: the input projection, every layer's mixer tensors with
/// the output norm's weight last (kept where the model does not run the
/// norm, as the host trainer's tensor order keeps it), every layer's MLP,
/// the final norm.
fn mimo_arena(w: &Mamba3MimoWeights) -> Vec<f32> {
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
            &l.out_proj_w,
            &l.norm_gate_weight,
        ] {
            out.extend_from_slice(t);
        }
    }
    for m in &w.mlp {
        for t in [&m.norm_weight, &m.fc1_w, &m.fc2_w] {
            out.extend_from_slice(t);
        }
    }
    out.extend_from_slice(&w.norm_f_weight);
    out
}

/// The control channels of a MIMO arena, as in SISO: the last
/// 3 * nheads + angles columns of every layer's in_proj gradient, then
/// dt_bias.
fn mimo_control(cfg: &Mamba3MimoConfig, w: &Mamba3MimoWeights) -> Vec<bool> {
    let (dm, ip) = (cfg.d_model, cfg.in_proj_out_dim());
    let ctrl = 3 * cfg.nheads() + cfg.num_rope_angles();
    let mut mask = vec![false; mimo_arena(w).len()];
    let mut at = w.input_proj_w.len() + w.input_proj_b.len();
    for l in &w.layers {
        let in_proj = at + l.norm_weight.len();
        mark_stripe(&mut mask, in_proj + ip - ctrl, dm, ctrl, ip);
        let dt_bias = in_proj + l.in_proj_w.len();
        mask[dt_bias..dt_bias + l.dt_bias.len()].fill(true);
        at = dt_bias
            + [
                &l.dt_bias,
                &l.b_norm_weight,
                &l.c_norm_weight,
                &l.b_bias,
                &l.c_bias,
                &l.d_param,
                &l.mimo_x,
                &l.mimo_z,
                &l.mimo_o,
                &l.out_proj_w,
                &l.norm_gate_weight,
            ]
            .iter()
            .map(|t| t.len())
            .sum::<usize>();
    }
    mask
}

fn sum_sq(g: &[f32], mask: &[bool], want: bool) -> f64 {
    g.iter()
        .zip(mask)
        .filter(|(_, m)| **m == want)
        .map(|(v, _)| f64::from(*v).powi(2))
        .sum()
}

/// One window's gradient before and after a step that clips the control
/// channels to a tenth of their norm, and the norm the step reports.
fn check(label: &str, before: &[f32], after: &[f32], mask: &[bool], bound: f32, reported: f32) {
    assert_eq!(before.len(), mask.len(), "{label}: arena length");
    let ctrl = sum_sq(before, mask, true).sqrt();
    let coef = f64::from(bound) / (ctrl + 1e-6);
    let mut scaled = 0usize;
    for (i, ((b, a), m)) in before.iter().zip(after).zip(mask).enumerate() {
        if *m {
            let want = f64::from(*b) * coef;
            assert!(
                (f64::from(*a) - want).abs() <= 1e-6 * want.abs() + 1e-30,
                "{label}: control element {i} is {a}, want {want}"
            );
            scaled += usize::from(*b != 0.0);
        } else {
            assert_eq!(a.to_bits(), b.to_bits(), "{label}: element {i} moved");
        }
    }
    assert!(
        scaled > 0,
        "{label}: the control channels carry no gradient"
    );
    let norm = (sum_sq(after, mask, true) + sum_sq(after, mask, false)).sqrt();
    let rel = (f64::from(reported) - norm).abs() / norm;
    eprintln!("{label}: control norm {ctrl:.4e} clipped to {bound:.4e}, reported {reported:.6e}");
    assert!(rel < 1e-5, "{label}: reported norm {reported} vs {norm}");
}

#[test]
fn control_clip_scales_only_the_control_channels() {
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        let cfg = Mamba3Config {
            d_model: 64,
            d_state: 16,
            headdim: 16,
            n_layers: 2,
            ..Mamba3Config::default()
        };
        let dm = cfg.d_model;
        let w = Mamba3Weights::init(&cfg, dm, 3);
        let mut t = Mamba3GpuTrainer::new(&w, cfg, session(dm), GpuOptions::default().dtype(dtype))
            .unwrap();
        let mut out = vec![0f32; BATCH * SEQ_LEN * dm];
        t.forward(&det(out.len(), 1), &mut out).unwrap();
        t.backward_step(
            &det(out.len(), 2),
            BackwardOpts::default().with_accumulate_only(true),
        )
        .unwrap();
        let stream = t.ctx().stream.clone();
        let before = t.grad_arena().to_cpu(&stream).unwrap();
        let mask = siso_control(&cfg, before.len());
        let bound = 0.1 * sum_sq(&before, &mask, true).sqrt() as f32;
        let m = t
            .apply_step_full(Some(f32::INFINITY), None, Some(bound))
            .unwrap();
        let after = t.grad_arena().to_cpu(&stream).unwrap();
        check(
            &format!("Mamba-3 SISO {dtype:?}"),
            &before,
            &after,
            &mask,
            bound,
            m.grad_norm.unwrap(),
        );

        let cfg = MambaConfig {
            d_model: 64,
            d_state: 16,
            n_layers: 2,
            ..MambaConfig::default()
        };
        let w = MambaWeights::init(&cfg, dm, 3);
        let mut t =
            MambaGpuTrainer::new(&w, cfg, session(dm), GpuOptions::default().dtype(dtype)).unwrap();
        t.forward(&det(out.len(), 1), &mut out).unwrap();
        t.backward_step(
            &det(out.len(), 2),
            BackwardOpts::default().with_accumulate_only(true),
        )
        .unwrap();
        let stream = t.ctx().stream.clone();
        let before = t.grad_arena().to_cpu(&stream).unwrap();
        let mask = m1_control(&cfg, before.len());
        let bound = 0.1 * sum_sq(&before, &mask, true).sqrt() as f32;
        let m = t
            .apply_step_full(Some(f32::INFINITY), None, Some(bound))
            .unwrap();
        let after = t.grad_arena().to_cpu(&stream).unwrap();
        check(
            &format!("Mamba-1 {dtype:?}"),
            &before,
            &after,
            &mask,
            bound,
            m.grad_norm.unwrap(),
        );
    }
}

/// Mamba-2 and MIMO clip the same way through their split seam: the
/// accumulating call leaves the window's gradient in the arena, and the
/// optimizer tail clips the control channels of the arena it steps on.
/// Mamba-2 with and without the in_proj bias (whose dt entries join the
/// region), MIMO with and without the output norm (which changes the
/// layer stride).
#[test]
fn control_clip_scales_only_the_control_channels_of_mamba2_and_mimo() {
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        let dm = 64;
        let mut out = vec![0f32; BATCH * SEQ_LEN * dm];
        for proj_bias in [false, true] {
            let cfg = Mamba2Config {
                d_model: dm,
                d_state: 16,
                headdim: 16,
                n_layers: 2,
                proj_bias,
                ..Mamba2Config::default()
            };
            let w = Mamba2Weights::init(&cfg, dm, 3);
            let mut t =
                Mamba2GpuTrainer::new(&w, cfg, session(dm), GpuOptions::default().dtype(dtype))
                    .unwrap();
            t.forward(&det(out.len(), 1), &mut out).unwrap();
            t.backward_step(
                &det(out.len(), 2),
                BackwardOpts::default().with_accumulate_only(true),
            )
            .unwrap();
            let stream = t.ctx().stream.clone();
            let before = t.grad_arena().unwrap().to_cpu(&stream).unwrap();
            let mask = m2_control(&cfg, &w);
            let bound = 0.1 * sum_sq(&before, &mask, true).sqrt() as f32;
            let m = t
                .apply_step_full(Some(f32::INFINITY), None, Some(bound))
                .unwrap();
            check(
                &format!("Mamba-2 {dtype:?} proj_bias={proj_bias}"),
                &before,
                &m2_arena(&t.grads().unwrap()),
                &mask,
                bound,
                m.grad_norm.unwrap(),
            );
        }
        for is_outproj_norm in [false, true] {
            let cfg = Mamba3MimoConfig {
                d_model: dm,
                d_state: 16,
                headdim: 16,
                n_layers: 2,
                is_outproj_norm,
                ..Mamba3MimoConfig::default()
            };
            let w = Mamba3MimoWeights::init(&cfg, dm, 3);
            let mut t =
                Mamba3MimoGpuTrainer::new(&w, cfg, session(dm), GpuOptions::default().dtype(dtype))
                    .unwrap();
            t.forward(&det(out.len(), 1), &mut out).unwrap();
            t.backward_step(
                &det(out.len(), 2),
                BackwardOpts::default().with_accumulate_only(true),
            )
            .unwrap();
            let stream = t.ctx().stream.clone();
            let before = t.grad_arena().unwrap().to_cpu(&stream).unwrap();
            let mask = mimo_control(&cfg, &w);
            let bound = 0.1 * sum_sq(&before, &mask, true).sqrt() as f32;
            let m = t
                .apply_step_full(Some(f32::INFINITY), None, Some(bound))
                .unwrap();
            check(
                &format!("MIMO {dtype:?} is_outproj_norm={is_outproj_norm}"),
                &before,
                &mimo_arena(&t.grads().unwrap()),
                &mask,
                bound,
                m.grad_norm.unwrap(),
            );
        }
    }
}

/// A clip of either kind on an accumulating call is refused in every
/// family: its norm is the whole window's.
#[test]
fn a_clip_on_an_accumulating_call_is_refused_in_every_family() {
    let dm = 32;
    let n = BATCH * SEQ_LEN * dm;
    let mut out = vec![0f32; n];
    let accumulating = BackwardOpts::default().with_accumulate_only(true);
    for opts in [
        accumulating.with_clip_max_norm(1.0),
        accumulating.with_control_clip_max_norm(1.0),
    ] {
        let cfg = MambaConfig {
            d_model: dm,
            n_layers: 1,
            ..MambaConfig::default()
        };
        let mut t = MambaGpuTrainer::new(
            &MambaWeights::init(&cfg, dm, 3),
            cfg,
            session(dm),
            GpuOptions::default(),
        )
        .unwrap();
        t.forward(&det(n, 1), &mut out).unwrap();
        assert!(t.backward_step(&det(n, 2), opts).is_err(), "Mamba-1");
        // The refusal leaves the forward pending, so a valid call follows.
        t.backward_step(&det(n, 2), accumulating).unwrap();

        let cfg = Mamba3Config {
            d_model: dm,
            headdim: 16,
            n_layers: 1,
            ..Mamba3Config::default()
        };
        let mut t = Mamba3GpuTrainer::new(
            &Mamba3Weights::init(&cfg, dm, 3),
            cfg,
            session(dm),
            GpuOptions::default(),
        )
        .unwrap();
        t.forward(&det(n, 1), &mut out).unwrap();
        assert!(t.backward_step(&det(n, 2), opts).is_err(), "SISO");
        t.backward_step(&det(n, 2), accumulating).unwrap();

        let cfg = Mamba2Config {
            d_model: dm,
            headdim: 16,
            n_layers: 1,
            ..Mamba2Config::default()
        };
        let mut t = Mamba2GpuTrainer::new(
            &Mamba2Weights::init(&cfg, dm, 3),
            cfg,
            session(dm),
            GpuOptions::default(),
        )
        .unwrap();
        t.forward(&det(n, 1), &mut out).unwrap();
        assert!(t.backward_step(&det(n, 2), opts).is_err(), "Mamba-2");

        let cfg = Mamba3MimoConfig {
            d_model: dm,
            headdim: 16,
            n_layers: 1,
            ..Mamba3MimoConfig::default()
        };
        let mut t = Mamba3MimoGpuTrainer::new(
            &Mamba3MimoWeights::init(&cfg, dm, 3),
            cfg,
            session(dm),
            GpuOptions::default(),
        )
        .unwrap();
        t.forward(&det(n, 1), &mut out).unwrap();
        assert!(t.backward_step(&det(n, 2), opts).is_err(), "MIMO");
    }
}
