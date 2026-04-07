//! End-to-end example: initialize model, train it, save, load, and run inference.
//!
//! ```bash
//! cargo run --example train_and_infer
//! ```

use std::path::Path;

use mamba_rs::config::MambaConfig;
use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::{MambaBackbone, MambaWeights};

fn main() {
    let cfg = MambaConfig::default(); // d_model=128, 3 layers, 366K params
    let input_dim = cfg.d_model;
    let seq_len = 16;
    let lr = 1e-3_f32;
    let steps = 50;

    println!("mamba-rs: train -> save -> load -> inference");
    println!("=============================================");
    println!(
        "Config: d_model={}, layers={}, d_inner={}, params={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.d_inner(),
        MambaBackbone::init(cfg, input_dim, 42).param_count()
    );
    println!();

    // =====================================================================
    // Step 1: Initialize weights
    // =====================================================================
    let inf_weights = MambaWeights::init(&cfg, input_dim, 42);
    let mut tw = train_weights_from_inference(&inf_weights);

    let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let dc = cfg.d_conv;
    let dm = cfg.d_model;

    // Pre-allocate scratch (reused every step)
    let mut acts = MambaBackboneFlat::zeros(dims);
    let mut fwd_scratch = PhaseScratch::zeros(&dims);
    let mut bwd_scratch = BackwardPhaseScratch::zeros(&dims);

    // Synthetic training data: random input, target = zeros (regression)
    let input: Vec<f32> = (0..seq_len * input_dim)
        .map(|i| ((i * 7 + 13) % 100) as f32 * 0.01 - 0.5)
        .collect();

    // =====================================================================
    // Step 2: Training loop (SGD, MSE loss)
    // =====================================================================
    println!("Training ({steps} steps, lr={lr}, seq_len={seq_len}):");

    for step in 0..steps {
        // Precompute a_neg = -exp(a_log)
        let mut a_neg = vec![0.0f32; cfg.n_layers * di * ds];
        for (l, lw) in tw.layers.iter().enumerate() {
            for i in 0..di * ds {
                a_neg[l * di * ds + i] = -lw.a_log[i].exp();
            }
        }

        // Forward pass
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

        // MSE loss
        let loss: f32 = temporal.iter().map(|v| v * v).sum::<f32>() / (seq_len * dm) as f32;

        // Backward pass
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

        // SGD weight update
        sgd_update_all(&mut tw, &grads, lr);

        if step % 10 == 0 || step == steps - 1 {
            println!("  step {step:3}: loss = {loss:.6}");
        }
    }

    // =====================================================================
    // Step 3: Save trained weights
    // =====================================================================
    let save_path = std::env::temp_dir().join("mamba_rs_trained.safetensors");
    let trained_inf_weights = inference_weights_from_train(&tw, &inf_weights);
    mamba_rs::serialize::save(&save_path, &trained_inf_weights, &cfg, input_dim)
        .expect("save failed");
    println!("\nSaved to: {}", save_path.display());

    // =====================================================================
    // Step 4: Load and run inference
    // =====================================================================
    let (loaded_w, loaded_cfg, loaded_input_dim) =
        mamba_rs::serialize::load(Path::new(&save_path)).expect("load failed");
    let bb = MambaBackbone::from_weights(loaded_cfg, loaded_w).expect("from_weights failed");
    assert_eq!(bb.input_dim(), loaded_input_dim);
    println!(
        "Loaded: d_model={}, layers={}, params={}",
        loaded_cfg.d_model,
        loaded_cfg.n_layers,
        bb.param_count()
    );

    // Run inference step-by-step
    let mut inf_state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();
    let mut output = vec![0.0f32; dm];

    println!("\nInference (10 steps):");
    for step in 0..10 {
        let inp: Vec<f32> = (0..input_dim)
            .map(|i| ((step * input_dim + i) * 7 + 13) as f32 % 100.0 * 0.01 - 0.5)
            .collect();
        bb.forward_step(&inp, &mut output, &mut inf_state, &mut scratch);

        let out_norm: f32 = output.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("  step {step}: output L2 norm = {out_norm:.6}");
    }

    // Cleanup
    std::fs::remove_file(&save_path).ok();
    println!("\nDone.");
}

fn sgd_update_all(tw: &mut TrainMambaWeights, grads: &TrainMambaWeights, lr: f32) {
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
}

fn sgd_update(params: &mut [f32], grads: &[f32], lr: f32) {
    for (p, g) in params.iter_mut().zip(grads.iter()) {
        *p -= lr * g;
    }
}

fn train_weights_from_inference(w: &MambaWeights) -> TrainMambaWeights {
    TrainMambaWeights {
        input_proj_w: w.input_proj_w.clone(),
        input_proj_b: w.input_proj_b.clone(),
        layers: w
            .layers
            .iter()
            .map(|lw| TrainMambaLayerWeights {
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
        norm_f_weight: w.norm_f_weight.clone(),
    }
}

fn inference_weights_from_train(tw: &TrainMambaWeights, template: &MambaWeights) -> MambaWeights {
    MambaWeights {
        input_proj_w: tw.input_proj_w.clone(),
        input_proj_b: tw.input_proj_b.clone(),
        layers: tw
            .layers
            .iter()
            .zip(template.layers.iter())
            .map(|(tlw, _)| mamba_rs::weights::MambaLayerWeights {
                norm_weight: tlw.norm_weight.clone(),
                in_proj_w: tlw.in_proj_w.clone(),
                conv1d_weight: tlw.conv1d_weight.clone(),
                conv1d_bias: tlw.conv1d_bias.clone(),
                x_proj_w: tlw.x_proj_w.clone(),
                dt_proj_w: tlw.dt_proj_w.clone(),
                dt_proj_b: tlw.dt_proj_b.clone(),
                a_log: tlw.a_log.clone(),
                a_neg: tlw.a_log.iter().map(|v| -v.exp()).collect(),
                d_param: tlw.d_param.clone(),
                out_proj_w: tlw.out_proj_w.clone(),
            })
            .collect(),
        norm_f_weight: tw.norm_f_weight.clone(),
    }
}
