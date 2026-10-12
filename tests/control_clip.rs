//! The control-channel clip bounds the gradients of the step size and the
//! decay on their own, before the global clip. After a step asked for it,
//! the arena holds the window's gradients with exactly those channels
//! scaled down to the bound and every other element untouched, and the
//! reported norm is that arena's. The test finds the channels from each
//! family's gradient layout by itself, so a region that drifts from the
//! layout fails here.
#![cfg(feature = "cuda")]

use mamba_rs::{
    BackwardOpts, GpuOptions, Mamba3Config, Mamba3GpuTrainer, Mamba3Weights, MambaConfig,
    MambaGpuTrainer, MambaWeights, TrainSessionCfg, WeightDtype,
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
