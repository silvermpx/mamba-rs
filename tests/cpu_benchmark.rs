use std::time::Instant;

use mamba_rs::MambaBackbone;
use mamba_rs::config::MambaConfig;
use mamba_rs::mamba_ssm::cpu::parallel::{parallel_mamba_backward, parallel_mamba_forward};
use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::weights::MambaWeights;

fn configs() -> [(&'static str, MambaConfig); 4] {
    [
        (
            "small",
            MambaConfig {
                d_model: 64,
                d_state: 16,
                d_conv: 4,
                expand: 2,
                n_layers: 2,
                ..Default::default()
            },
        ),
        ("default", MambaConfig::default()),
        (
            "medium",
            MambaConfig {
                d_model: 256,
                d_state: 16,
                d_conv: 4,
                expand: 2,
                n_layers: 4,
                ..Default::default()
            },
        ),
        (
            "large",
            MambaConfig {
                d_model: 512,
                d_state: 16,
                d_conv: 4,
                expand: 2,
                n_layers: 6,
                ..Default::default()
            },
        ),
    ]
}

/// Full CPU benchmark: inference first, then training.
///
/// Run: `cargo test --release --test cpu_benchmark -- --ignored --nocapture`
#[test]
#[ignore]
fn cpu_benchmark() {
    // ===================================================================
    // Part 1: Inference (T=1 step)
    // ===================================================================
    println!("mamba-rs inference benchmark (T=1)");
    println!("===================================");
    println!();

    for (name, cfg) in &configs() {
        let input_dim = cfg.d_model;
        let backbone = MambaBackbone::init(*cfg, input_dim, 42);
        let mut state = backbone.alloc_state();
        let mut scratch = backbone.alloc_scratch();
        let mut output = vec![0.0f32; cfg.d_model];
        let input = vec![0.1f32; input_dim];

        // warmup
        for _ in 0..100 {
            backbone.forward_step(&input, &mut output, &mut state, &mut scratch);
        }
        state.reset();

        // bench
        let iterations = 10_000;
        let t0 = Instant::now();
        for _ in 0..iterations {
            backbone.forward_step(&input, &mut output, &mut state, &mut scratch);
        }
        let us_per_step = t0.elapsed().as_micros() as f64 / iterations as f64;

        println!(
            "{name:>8}: d_model={:>3}, layers={}, d_inner={:>4}, params={:>7} | {:.1} us/step",
            cfg.d_model,
            cfg.n_layers,
            cfg.d_inner(),
            backbone.param_count(),
            us_per_step,
        );
    }

    println!();

    // ===================================================================
    // Part 2: Training forward + backward (B=1, T=32)
    // ===================================================================
    let seq_len = 32;
    println!("mamba-rs CPU training benchmark (B=1, T={seq_len})");
    println!("===================================================");
    println!();

    for (name, cfg) in &configs() {
        let input_dim = cfg.d_model;
        let dims = MambaDims::from_config(cfg, seq_len, input_dim);
        let di = dims.d_inner;
        let ds = dims.d_state;

        let inf_w = MambaWeights::init(cfg, input_dim, 42);
        let tw = TrainMambaWeights {
            input_proj_w: inf_w.input_proj_w.clone(),
            input_proj_b: inf_w.input_proj_b.clone(),
            layers: inf_w
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
            norm_f_weight: inf_w.norm_f_weight.clone(),
        };

        let mut a_neg = vec![0.0f32; cfg.n_layers * di * ds];
        for (l, lw) in tw.layers.iter().enumerate() {
            for i in 0..di * ds {
                a_neg[l * di * ds + i] = -lw.a_log[i].exp();
            }
        }

        let input_data: Vec<f32> = (0..seq_len * input_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();

        // Pre-allocate scratch (reused every iteration)
        let mut acts = MambaBackboneFlat::zeros(dims);
        let mut fwd_scratch = PhaseScratch::zeros(&dims);
        let mut bwd_scratch = BackwardPhaseScratch::zeros(&dims);
        let mut temporal = vec![0.0f32; seq_len * cfg.d_model];
        let mut d_temporal = vec![0.0f32; seq_len * cfg.d_model];
        let mut grads = TrainMambaWeights::zeros_from_dims(&dims);

        // --- Forward only ---
        let iters = if cfg.d_model >= 512 { 100 } else { 500 };
        for _ in 0..20 {
            let mut conv = vec![0.0f32; cfg.n_layers * di * cfg.d_conv];
            let mut ssm = vec![0.0f32; cfg.n_layers * di * ds];
            let mut state = MambaRecurrentState {
                conv: &mut conv,
                ssm: &mut ssm,
                a_neg: &a_neg,
            };
            forward_mamba_backbone_batched(
                &mut temporal,
                &mut acts,
                &tw,
                &input_data,
                &mut state,
                &mut fwd_scratch,
                &dims,
            );
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            let mut conv = vec![0.0f32; cfg.n_layers * di * cfg.d_conv];
            let mut ssm = vec![0.0f32; cfg.n_layers * di * ds];
            let mut state = MambaRecurrentState {
                conv: &mut conv,
                ssm: &mut ssm,
                a_neg: &a_neg,
            };
            forward_mamba_backbone_batched(
                &mut temporal,
                &mut acts,
                &tw,
                &input_data,
                &mut state,
                &mut fwd_scratch,
                &dims,
            );
        }
        let fwd_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // --- Forward + Backward ---
        let iters = if cfg.d_model >= 512 { 50 } else { 200 };
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut conv = vec![0.0f32; cfg.n_layers * di * cfg.d_conv];
            let mut ssm = vec![0.0f32; cfg.n_layers * di * ds];
            let mut state = MambaRecurrentState {
                conv: &mut conv,
                ssm: &mut ssm,
                a_neg: &a_neg,
            };
            forward_mamba_backbone_batched(
                &mut temporal,
                &mut acts,
                &tw,
                &input_data,
                &mut state,
                &mut fwd_scratch,
                &dims,
            );

            d_temporal.fill(1.0);
            grads.zero();
            backward_mamba_backbone_batched(
                &mut d_temporal,
                &mut grads,
                &acts,
                &tw,
                &a_neg,
                &mut bwd_scratch,
                &dims,
            );
        }
        let fwdbwd_us = t0.elapsed().as_micros() as f64 / iters as f64;
        let bwd_us = fwdbwd_us - fwd_us;

        let params = MambaBackbone::init(*cfg, input_dim, 42).param_count();
        println!(
            "{name:>8}: d_model={:>3}, layers={}, params={:>7} | fwd {fwd_us:>8.1} us | bwd {bwd_us:>8.1} us | total {fwdbwd_us:>8.1} us",
            cfg.d_model, cfg.n_layers, params,
        );
    }
    println!();

    // ===================================================================
    // Part 3: Parallel training (B=batch, T=32, all cores)
    // ===================================================================
    let batch_sizes = [16, 64, 128];
    let cfg = MambaConfig::default(); // 128, 3L, 366K
    let input_dim = cfg.d_model;

    println!(
        "mamba-rs parallel training (d_model={}, layers={}, T={seq_len}, {} rayon threads)",
        cfg.d_model,
        cfg.n_layers,
        rayon::current_num_threads()
    );
    println!("========================================================================");
    println!();

    for &b_sz in &batch_sizes {
        let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
        let di = dims.d_inner;
        let ds = dims.d_state;

        let inf_w = MambaWeights::init(&cfg, input_dim, 42);
        let tw = TrainMambaWeights {
            input_proj_w: inf_w.input_proj_w.clone(),
            input_proj_b: inf_w.input_proj_b.clone(),
            layers: inf_w
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
            norm_f_weight: inf_w.norm_f_weight.clone(),
        };

        let mut a_neg = vec![0.0f32; cfg.n_layers * di * ds];
        for (l, lw) in tw.layers.iter().enumerate() {
            for i in 0..di * ds {
                a_neg[l * di * ds + i] = -lw.a_log[i].exp();
            }
        }

        let mamba_inputs: Vec<f32> = (0..b_sz * seq_len * input_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();

        // --- Parallel Forward ---
        let iters = if b_sz >= 128 { 30 } else { 100 };
        // Warmup
        for _ in 0..5 {
            let mut temporal = vec![0.0f32; b_sz * seq_len * cfg.d_model];
            let mut acts: Vec<MambaBackboneFlat> =
                (0..b_sz).map(|_| MambaBackboneFlat::zeros(dims)).collect();
            let mut conv = vec![0.0f32; b_sz * cfg.n_layers * di * cfg.d_conv];
            let mut ssm = vec![0.0f32; b_sz * cfg.n_layers * di * ds];
            let mut state = MambaRecurrentState {
                conv: &mut conv,
                ssm: &mut ssm,
                a_neg: &a_neg,
            };
            parallel_mamba_forward(
                &mut temporal,
                &mut acts,
                &mut state,
                &tw,
                &mamba_inputs,
                &dims,
                b_sz,
            );
        }

        let t0 = Instant::now();
        let mut acts_bench: Vec<MambaBackboneFlat> =
            (0..b_sz).map(|_| MambaBackboneFlat::zeros(dims)).collect();
        for _ in 0..iters {
            let mut temporal = vec![0.0f32; b_sz * seq_len * cfg.d_model];
            let mut conv = vec![0.0f32; b_sz * cfg.n_layers * di * cfg.d_conv];
            let mut ssm = vec![0.0f32; b_sz * cfg.n_layers * di * ds];
            let mut state = MambaRecurrentState {
                conv: &mut conv,
                ssm: &mut ssm,
                a_neg: &a_neg,
            };
            parallel_mamba_forward(
                &mut temporal,
                &mut acts_bench,
                &mut state,
                &tw,
                &mamba_inputs,
                &dims,
                b_sz,
            );
        }
        let par_fwd_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // --- Parallel Forward + Backward ---
        let iters = if b_sz >= 128 { 20 } else { 50 };
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut temporal = vec![0.0f32; b_sz * seq_len * cfg.d_model];
            let mut conv = vec![0.0f32; b_sz * cfg.n_layers * di * cfg.d_conv];
            let mut ssm = vec![0.0f32; b_sz * cfg.n_layers * di * ds];
            let mut state = MambaRecurrentState {
                conv: &mut conv,
                ssm: &mut ssm,
                a_neg: &a_neg,
            };
            parallel_mamba_forward(
                &mut temporal,
                &mut acts_bench,
                &mut state,
                &tw,
                &mamba_inputs,
                &dims,
                b_sz,
            );

            let dm = cfg.d_model;
            let mut d_temporal_seqs: Vec<Vec<f32>> =
                (0..b_sz).map(|_| vec![1.0f32; seq_len * dm]).collect();
            let mut grads = TrainMambaWeights::zeros_from_dims(&dims);
            parallel_mamba_backward(
                &mut d_temporal_seqs,
                &mut grads,
                &acts_bench,
                &tw,
                &a_neg,
                &dims,
            );
        }
        let par_fwdbwd_us = t0.elapsed().as_micros() as f64 / iters as f64;
        let par_bwd_us = par_fwdbwd_us - par_fwd_us;
        let samples_per_sec = b_sz as f64 / (par_fwdbwd_us / 1_000_000.0);

        println!(
            "  B={b_sz:>3}: fwd {par_fwd_us:>10.1} us | bwd {par_bwd_us:>10.1} us | total {par_fwdbwd_us:>10.1} us | {samples_per_sec:>8.0} samples/sec",
        );
    }
    println!();
}
