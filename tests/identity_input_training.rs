//! A model without an input projection trains on every lane: the official
//! checkpoints feed the embedding straight into the first norm, and their
//! weights carry no projection. The forward and backward of such a model
//! are those of an explicit identity projection, so one f32 step moves
//! every other parameter exactly as the model with that projection moves
//! it, for Mamba-3 SISO and for Mamba-1.
#![cfg(feature = "cuda")]

use mamba_rs::{
    GpuOptions, Mamba3Config, Mamba3GpuTrainer, Mamba3Weights, MambaConfig, MambaGpuTrainer,
    MambaWeights, TrainSessionCfg,
};

const BATCH: usize = 2;
const SEQ_LEN: usize = 40;

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.5
        })
        .collect()
}

/// Every trained tensor but the input projection, in a fixed order.
fn body(w: &Mamba3Weights) -> Vec<u32> {
    let mut v = Vec::new();
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
            &l.norm_gate_weight,
            &l.out_proj_w,
        ] {
            v.extend(t.iter().map(|x| x.to_bits()));
        }
    }
    v.extend(w.norm_f_weight.iter().map(|x| x.to_bits()));
    v
}

#[test]
fn siso_f32_identity_input_trains_like_an_explicit_identity() {
    for (is_outproj_norm, parallel) in [(true, true), (false, false)] {
        let cfg = Mamba3Config {
            d_model: 32,
            d_state: 16,
            expand: 2,
            headdim: 16,
            n_layers: 2,
            is_outproj_norm,
            scan_mode: if parallel {
                mamba_rs::config::ScanMode::Auto
            } else {
                mamba_rs::config::ScanMode::Sequential
            },
            ..Mamba3Config::default()
        };
        let dm = cfg.d_model;
        let mut explicit = Mamba3Weights::init(&cfg, dm, 11);
        explicit.input_proj_w = (0..dm * dm)
            .map(|i| if i / dm == i % dm { 1.0 } else { 0.0 })
            .collect();
        explicit.input_proj_b = vec![0.0; dm];
        let mut identity = explicit.clone();
        identity.input_proj_w.clear();
        identity.input_proj_b.clear();

        let session = TrainSessionCfg {
            input_dim: dm,
            batch: BATCH,
            seq_len: SEQ_LEN,
            lr: 1e-3,
            weight_decay: 0.1,
        };
        let input = values(BATCH * SEQ_LEN * dm, 1);
        let d_out = values(BATCH * SEQ_LEN * dm, 2);
        let mut stepped = Vec::new();
        for w in [&explicit, &identity] {
            let mut t = Mamba3GpuTrainer::new(w, cfg, session, GpuOptions::default()).unwrap();
            t.step(&input, &d_out).unwrap();
            stepped.push(body(&t.snapshot_master().unwrap()));
        }
        assert_eq!(
            stepped[0], stepped[1],
            "outproj_norm {is_outproj_norm} parallel {parallel}: the identity model moved \
             differently from the explicit identity projection"
        );
    }
}

/// Every trained tensor of a Mamba-1 model but its input projection.
fn m1_body(w: &MambaWeights) -> Vec<u32> {
    let mut v = Vec::new();
    for l in &w.layers {
        for t in [
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
            v.extend(t.iter().map(|x| x.to_bits()));
        }
    }
    v.extend(w.norm_f_weight.iter().map(|x| x.to_bits()));
    v
}

#[test]
fn m1_f32_identity_input_trains_like_an_explicit_identity() {
    for scan_mode in [
        mamba_rs::config::ScanMode::Auto,
        mamba_rs::config::ScanMode::Sequential,
    ] {
        let cfg = MambaConfig {
            d_model: 32,
            n_layers: 2,
            d_state: 16,
            scan_mode,
            ..MambaConfig::default()
        };
        let dm = cfg.d_model;
        let mut explicit = MambaWeights::init(&cfg, dm, 11);
        explicit.input_proj_w = (0..dm * dm)
            .map(|i| if i / dm == i % dm { 1.0 } else { 0.0 })
            .collect();
        explicit.input_proj_b = vec![0.0; dm];
        let mut identity = explicit.clone();
        identity.input_proj_w.clear();
        identity.input_proj_b.clear();

        let session = TrainSessionCfg {
            input_dim: dm,
            batch: BATCH,
            seq_len: SEQ_LEN,
            lr: 1e-3,
            weight_decay: 0.1,
        };
        let input = values(BATCH * SEQ_LEN * dm, 1);
        let d_out = values(BATCH * SEQ_LEN * dm, 2);
        let mut stepped = Vec::new();
        for w in [&explicit, &identity] {
            let mut t = MambaGpuTrainer::new(w, cfg, session, GpuOptions::default()).unwrap();
            t.step(&input, &d_out).unwrap();
            stepped.push(m1_body(&t.snapshot_master().unwrap()));
        }
        assert_eq!(
            stepped[0], stepped[1],
            "{scan_mode:?}: the identity model moved differently from the explicit identity \
             projection"
        );
    }
}
