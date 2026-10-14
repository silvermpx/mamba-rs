//! A Mamba-1 model without an input projection, the layout of the official
//! checkpoints, runs every CPU path: the training forward and backward,
//! the single-step target forward and batched decode. Each gives, bit for
//! bit, what the same model with an explicit identity projection gives,
//! and the backbone's batched step is its single step on every sample,
//! packed projections included.

use mamba_rs::inference::{MambaStepScratch, mamba_step_batch};
use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::target::{MambaTargetScratch, forward_mamba_target_step};
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::{MambaBackbone, MambaConfig, MambaState, MambaWeights};

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.5
        })
        .collect()
}

fn cfg() -> MambaConfig {
    MambaConfig {
        d_model: 32,
        n_layers: 2,
        d_state: 8,
        ..MambaConfig::default()
    }
}

/// The model with an explicit identity projection, and the same model
/// without one.
fn pair(cfg: &MambaConfig) -> (MambaWeights, MambaWeights) {
    let dm = cfg.d_model;
    let mut explicit = MambaWeights::init(cfg, dm, 9);
    explicit.input_proj_w = (0..dm * dm)
        .map(|i| if i / dm == i % dm { 1.0 } else { 0.0 })
        .collect();
    explicit.input_proj_b = vec![0.0; dm];
    let mut identity = explicit.clone();
    identity.input_proj_w.clear();
    identity.input_proj_b.clear();
    (explicit, identity)
}

fn train_weights(w: &MambaWeights) -> TrainMambaWeights {
    TrainMambaWeights {
        input_proj_w: w.input_proj_w.clone(),
        input_proj_b: w.input_proj_b.clone(),
        layers: w
            .layers
            .iter()
            .map(|l| TrainMambaLayerWeights {
                norm_weight: l.norm_weight.clone(),
                norm_bias: l.norm_bias.clone(),
                in_proj_w: l.in_proj_w.clone(),
                in_proj_b: l.in_proj_b.clone(),
                conv1d_weight: l.conv1d_weight.clone(),
                conv1d_bias: l.conv1d_bias.clone(),
                x_proj_w: l.x_proj_w.clone(),
                dt_proj_w: l.dt_proj_w.clone(),
                dt_proj_b: l.dt_proj_b.clone(),
                a_log: l.a_log.clone(),
                d_param: l.d_param.clone(),
                out_proj_w: l.out_proj_w.clone(),
                out_proj_b: l.out_proj_b.clone(),
            })
            .collect(),
        norm_f_weight: w.norm_f_weight.clone(),
        norm_f_bias: w.norm_f_bias.clone(),
        mlp: w.mlp.clone(),
    }
}

/// Every gradient but the input projection's, in a fixed order.
fn body_grads(g: &TrainMambaWeights) -> Vec<u32> {
    let mut v = Vec::new();
    for l in &g.layers {
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
    v.extend(g.norm_f_weight.iter().map(|x| x.to_bits()));
    v
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn m1_identity_input_trains_on_the_cpu_like_an_explicit_identity() {
    let cfg = cfg();
    let (explicit, identity) = pair(&cfg);
    let (dm, di, ds, dc, nl) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.d_state,
        cfg.d_conv,
        cfg.n_layers,
    );
    let seq_len = 12;
    let dims = MambaDims::from_config(&cfg, seq_len, dm);
    let input = values(seq_len * dm, 1);
    let d_out = values(seq_len * dm, 2);
    let mut runs = Vec::new();
    for w in [&explicit, &identity] {
        let tw = train_weights(w);
        let a_neg: Vec<f32> = w
            .layers
            .iter()
            .flat_map(|l| l.a_log.iter().map(|a| -a.exp()))
            .collect();
        let mut acts = MambaBackboneFlat::zeros(dims);
        let mut scratch = PhaseScratch::zeros(&dims);
        let mut conv = vec![0.0f32; nl * di * dc];
        let mut ssm = vec![0.0f32; nl * di * ds];
        let mut state = MambaRecurrentState {
            conv: &mut conv,
            ssm: &mut ssm,
            a_neg: &a_neg,
        };
        let mut temporal = vec![0.0f32; seq_len * dm];
        forward_mamba_backbone_batched(
            &mut temporal,
            &mut acts,
            &tw,
            &input,
            &mut state,
            &mut scratch,
            &dims,
        );
        let mut grads = TrainMambaWeights::zeros_from_dims(&dims);
        let mut d_temporal = d_out.clone();
        let mut bwd = BackwardPhaseScratch::zeros(&dims);
        backward_mamba_backbone_batched(
            &mut d_temporal,
            &mut grads,
            &acts,
            &tw,
            &a_neg,
            &mut bwd,
            &dims,
        );
        runs.push((bits(&temporal), body_grads(&grads)));
    }
    assert_eq!(runs[0].0, runs[1].0, "the training forward differs");
    assert_eq!(runs[0].1, runs[1].1, "the training gradients differ");

    let mut outs = Vec::new();
    for w in [&explicit, &identity] {
        let mut scratch = MambaTargetScratch::new(dm, di, ds, cfg.dt_rank());
        let mut out = vec![0.0f32; dm];
        forward_mamba_target_step(
            &mut out,
            &input[..dm],
            &train_weights(w),
            &mut scratch,
            (dm, di, ds, dc, cfg.dt_rank(), dm),
            (cfg.rms_norm_eps, cfg.bcdt_rms_eps),
        );
        outs.push(bits(&out));
    }
    assert_eq!(outs[0], outs[1], "the target step differs");
}

#[test]
fn m1_identity_input_decodes_in_batches() {
    const BATCH: usize = 5;
    let cfg = cfg();
    let dm = cfg.d_model;
    let (explicit, identity) = pair(&cfg);
    let inputs = values(BATCH * dm, 3);
    let mut batched = Vec::new();
    for w in [&explicit, &identity] {
        let mut states: Vec<MambaState> = (0..BATCH)
            .map(|_| MambaState::zeros(cfg.n_layers, cfg.d_inner(), cfg.d_state, cfg.d_conv))
            .collect();
        let mut scratches: Vec<MambaStepScratch> =
            (0..BATCH).map(|_| MambaStepScratch::new(&cfg)).collect();
        let mut out = vec![0.0f32; BATCH * dm];
        for _ in 0..3 {
            mamba_step_batch(&inputs, &mut out, w, &mut states, &mut scratches, &cfg, dm);
        }
        batched.push(bits(&out));
    }
    assert_eq!(batched[0], batched[1], "batched decode differs");

    for pack in [false, true] {
        let mut bb = MambaBackbone::new(identity.clone(), cfg, dm).unwrap();
        if pack {
            bb.pack_for_decode();
        }
        let mut states: Vec<MambaState> = (0..BATCH).map(|_| bb.alloc_state()).collect();
        let mut scratches: Vec<MambaStepScratch> = (0..BATCH).map(|_| bb.alloc_scratch()).collect();
        let mut out = vec![0.0f32; BATCH * dm];
        let mut single = vec![0.0f32; BATCH * dm];
        let mut one_states: Vec<MambaState> = (0..BATCH).map(|_| bb.alloc_state()).collect();
        let mut one_scratch = bb.alloc_scratch();
        for _ in 0..3 {
            bb.forward_step_batch(&inputs, &mut out, &mut states, &mut scratches);
            for (b, state) in one_states.iter_mut().enumerate() {
                bb.forward_step(
                    &inputs[b * dm..(b + 1) * dm],
                    &mut single[b * dm..(b + 1) * dm],
                    state,
                    &mut one_scratch,
                );
            }
        }
        assert_eq!(
            bits(&out),
            bits(&single),
            "packed {pack}: the batched step differs from the single step"
        );
    }
}
