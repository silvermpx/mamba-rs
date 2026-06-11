//! Mamba-3 SISO inference example.
//!
//! ```bash
//! cargo run --example mamba3_inference
//! ```

use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::inference::{Mamba3StepScratch, mamba3_step};
use mamba_rs::mamba3_siso::state::Mamba3State;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn main() {
    let cfg = Mamba3Config {
        d_model: 64,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 4,
        rope_fraction: 0.5,
        a_floor: 1e-4, // reference default (state-spaces/mamba A_floor)
        is_outproj_norm: false,
    };
    cfg.validate().unwrap();

    let input_dim = 32;
    let weights = Mamba3Weights::init(&cfg, input_dim, 42);
    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3StepScratch::new(&cfg);
    let mut temporal = vec![0.0_f32; cfg.d_model];

    println!(
        "Mamba-3 SISO: d_model={}, d_state={}, nheads={}, n_layers={}",
        cfg.d_model,
        cfg.d_state,
        cfg.nheads(),
        cfg.n_layers
    );
    println!(
        "  headdim={}, ngroups={}, rope_angles={}",
        cfg.headdim,
        cfg.ngroups,
        cfg.num_rope_angles()
    );
    println!("  in_proj_dim={}", cfg.in_proj_out_dim());

    let input = vec![1.0_f32; input_dim];

    for step in 0..10 {
        mamba3_step(
            &mut temporal,
            &input,
            &mut scratch,
            &weights,
            &mut state.layers,
            &cfg,
        );
        let norm: f32 = temporal.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("Step {step}: output norm = {norm:.6}");
    }

    println!("\nState after 10 steps:");
    let l0 = &state.layers[0];
    let ssm_norm: f32 = l0.ssm_state.iter().map(|v| v * v).sum::<f32>().sqrt();
    let angle_max: f32 = l0.angle_state.iter().cloned().fold(0.0, f32::max);
    println!("  Layer 0 SSM state norm: {ssm_norm:.6}");
    println!("  Layer 0 max angle: {angle_max:.4} (should be in [0, 2pi))");
}
