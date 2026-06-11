//! Mamba-3 SISO training forward+backward example.
//!
//! ```bash
//! cargo run --example mamba3_training
//! ```

use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::backward::backward_mamba3_layer_batched;
use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
use mamba_rs::mamba3_siso::cpu::flat::Mamba3LayerFlat;
use mamba_rs::mamba3_siso::cpu::forward::{Mamba3LayerStateMut, forward_mamba3_layer_batched};
use mamba_rs::mamba3_siso::cpu::scratch::Mamba3Scratch;
use mamba_rs::mamba3_siso::cpu::weights::TrainMamba3LayerWeights;

fn main() {
    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
    };
    cfg.validate();

    let seq_len = 16;
    let dims = Mamba3Dims::from_config(&cfg, seq_len);

    // Init weights with some nonzero values
    let mut w = TrainMamba3LayerWeights::zeros(&dims);
    for v in &mut w.norm_weight {
        *v = 1.0;
    }
    for v in &mut w.d_param {
        *v = 1.0;
    }
    for v in &mut w.b_norm_weight {
        *v = 1.0;
    }
    for v in &mut w.c_norm_weight {
        *v = 1.0;
    }
    for (i, v) in w.in_proj_w.iter_mut().enumerate() {
        *v = ((i % 7) as f32 - 3.0) * 0.01;
    }
    for (i, v) in w.out_proj_w.iter_mut().enumerate() {
        *v = ((i % 5) as f32 - 2.0) * 0.01;
    }

    let mut acts = Mamba3LayerFlat::zeros(dims);
    let mut scratch = Mamba3Scratch::zeros(&dims);
    let nh = dims.nheads;
    let hd = dims.headdim;
    let ds = dims.d_state;
    let na = dims.num_rope_angles.max(1);
    let mut ssm = vec![0.0; nh * hd * ds];
    let mut k_st = vec![0.0; nh * ds];
    let mut v_st = vec![0.0; nh * hd];
    let mut a_st = vec![0.0; nh * na];
    let mut temporal = vec![0.5_f32; seq_len * dims.d_model];

    println!(
        "Mamba-3 SISO Training: d_model={}, seq_len={}",
        dims.d_model, seq_len
    );
    println!("  nheads={}, headdim={}, d_state={}", nh, hd, ds);

    // Forward
    forward_mamba3_layer_batched(
        &mut temporal,
        &mut acts,
        &w,
        Mamba3LayerStateMut {
            ssm: &mut ssm,
            k: &mut k_st,
            v: &mut v_st,
            angle: &mut a_st,
        },
        &mut scratch,
        &dims,
    );

    let fwd_norm: f32 = temporal.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("Forward output norm: {fwd_norm:.6}");

    // Backward
    let mut d_temporal = vec![1.0_f32; seq_len * dims.d_model];
    let mut d_w = TrainMamba3LayerWeights::zeros(&dims);

    backward_mamba3_layer_batched(
        &mut d_temporal,
        &acts,
        &w,
        &mut d_w,
        &mut scratch,
        &dims,
        None,
    );

    let grad_norm = d_w.sum_sq().sqrt();
    println!("Gradient norm: {grad_norm:.6}");
    println!(
        "d_temporal norm: {:.6}",
        d_temporal
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt()
    );

    // Check all finite
    assert!(
        temporal.iter().all(|v| v.is_finite()),
        "forward output not finite"
    );
    assert!(
        d_temporal.iter().all(|v| v.is_finite()),
        "d_temporal not finite"
    );
    println!("\nAll outputs and gradients finite. Training pipeline works.");
}
