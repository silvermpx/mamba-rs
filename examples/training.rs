//! CPU training example: forward + backward + simple SGD update.
//!
//! ```bash
//! cargo run --example training
//! ```

use mamba_rs::MambaWeights;
use mamba_rs::config::MambaConfig;
use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::TrainMambaWeights;

fn main() {
    let cfg = MambaConfig::default();
    let input_dim = cfg.d_model;
    let seq_len = 16;
    let lr = 1e-3_f32;

    let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let dc = cfg.d_conv;

    // Initialize weights
    let inf_weights = MambaWeights::init(&cfg, input_dim, 42);
    let mut tw = TrainMambaWeights {
        input_proj_w: inf_weights.input_proj_w.clone(),
        input_proj_b: inf_weights.input_proj_b.clone(),
        layers: inf_weights
            .layers
            .iter()
            .map(|lw| mamba_rs::train::weights::TrainMambaLayerWeights {
                norm_weight: lw.norm_weight.clone(),
                in_proj_w: lw.in_proj_w.clone(),
                conv1d_weight: lw.conv1d_weight.clone(),
                conv1d_bias: lw.conv1d_bias.clone(),
                x_proj_w: lw.x_proj_w.clone(),
                dt_proj_w: lw.dt_proj_w.clone(),
                dt_proj_b: lw.dt_proj_b.clone(),
                a_log: lw.a_log.clone(),
                d_param: lw.d_param.clone(),
                out_proj_w: lw.out_proj_w.clone(),
            })
            .collect(),
        norm_f_weight: inf_weights.norm_f_weight.clone(),
    };

    // Allocate buffers
    let mut acts = MambaBackboneFlat::zeros(dims);
    let mut fwd_scratch = PhaseScratch::zeros(&dims);
    let mut bwd_scratch = BackwardPhaseScratch::zeros(&dims);

    // Random input sequence
    let input: Vec<f32> = (0..seq_len * input_dim)
        .map(|i| ((i * 7 + 13) % 100) as f32 * 0.01 - 0.5)
        .collect();

    // Target: zeros (simple regression loss)
    let dm = cfg.d_model;

    println!(
        "Training: {} layers, d_model={}, seq_len={}, {} params",
        cfg.n_layers,
        dm,
        seq_len,
        inf_weights.input_proj_w.len()
            + inf_weights.input_proj_b.len()
            + inf_weights
                .layers
                .iter()
                .map(|l| {
                    l.norm_weight.len()
                        + l.in_proj_w.len()
                        + l.conv1d_weight.len()
                        + l.conv1d_bias.len()
                        + l.x_proj_w.len()
                        + l.dt_proj_w.len()
                        + l.dt_proj_b.len()
                        + l.a_log.len()
                        + l.d_param.len()
                        + l.out_proj_w.len()
                })
                .sum::<usize>()
            + inf_weights.norm_f_weight.len()
    );

    // Training loop
    for step in 0..20 {
        // Precompute a_neg
        let mut a_neg = vec![0.0f32; cfg.n_layers * di * ds];
        for (l, lw) in tw.layers.iter().enumerate() {
            for i in 0..di * ds {
                a_neg[l * di * ds + i] = -lw.a_log[i].exp();
            }
        }

        // Forward
        let mut conv = vec![0.0f32; cfg.n_layers * di * dc];
        let mut ssm = vec![0.0f32; cfg.n_layers * di * ds];
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
            &mut fwd_scratch,
            &dims,
        );

        // Loss = mean(temporal^2) — regression to zero
        let loss: f32 = temporal.iter().map(|v| v * v).sum::<f32>() / (seq_len * dm) as f32;

        // Backward: d_loss/d_temporal = 2 * temporal / (seq_len * dm)
        let scale = 2.0 / (seq_len * dm) as f32;
        let mut d_temporal: Vec<f32> = temporal.iter().map(|v| v * scale).collect();

        let mut grads = TrainMambaWeights::zeros_from_dims(&dims);
        backward_mamba_backbone_batched(
            &mut d_temporal,
            &mut grads,
            &acts,
            &tw,
            &a_neg,
            &mut bwd_scratch,
            &dims,
        );

        // SGD update
        sgd_update(&mut tw.input_proj_w, &grads.input_proj_w, lr);
        sgd_update(&mut tw.input_proj_b, &grads.input_proj_b, lr);
        sgd_update(&mut tw.norm_f_weight, &grads.norm_f_weight, lr);
        for (lw, gw) in tw.layers.iter_mut().zip(grads.layers.iter()) {
            sgd_update(&mut lw.norm_weight, &gw.norm_weight, lr);
            sgd_update(&mut lw.in_proj_w, &gw.in_proj_w, lr);
            sgd_update(&mut lw.conv1d_weight, &gw.conv1d_weight, lr);
            sgd_update(&mut lw.conv1d_bias, &gw.conv1d_bias, lr);
            sgd_update(&mut lw.x_proj_w, &gw.x_proj_w, lr);
            sgd_update(&mut lw.dt_proj_w, &gw.dt_proj_w, lr);
            sgd_update(&mut lw.dt_proj_b, &gw.dt_proj_b, lr);
            sgd_update(&mut lw.a_log, &gw.a_log, lr);
            sgd_update(&mut lw.d_param, &gw.d_param, lr);
            sgd_update(&mut lw.out_proj_w, &gw.out_proj_w, lr);
        }

        if step % 5 == 0 {
            let grad_norm: f32 = grads.input_proj_w.iter().map(|v| v * v).sum::<f32>().sqrt();
            println!("step {step:3}: loss = {loss:.6}, grad_norm(input_proj) = {grad_norm:.6}");
        }
    }
}

fn sgd_update(params: &mut [f32], grads: &[f32], lr: f32) {
    for (p, g) in params.iter_mut().zip(grads.iter()) {
        *p -= lr * g;
    }
}
