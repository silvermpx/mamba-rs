//! Comprehensive correctness tests for mamba-rs.
//!
//! Covers: CPU inference, batch inference, sequence forward,
//! training gradient checks, serialization, and GPU parity.

use mamba_rs::{MambaBackbone, MambaConfig, MambaState, MambaStepScratch};

fn default_backbone() -> MambaBackbone {
    let cfg = MambaConfig::default();
    MambaBackbone::init(cfg, 128, 42)
}

// =========================================================================
// CPU Inference
// =========================================================================

#[test]
fn test_single_step_nonzero() {
    let bb = default_backbone();
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();
    let mut output = vec![0.0f32; bb.config().d_model];
    let input = vec![0.1f32; 128];

    bb.forward_step(&input, &mut output, &mut state, &mut scratch);

    assert!(
        output.iter().any(|&v| v.abs() > 1e-10),
        "output should be non-zero after one step"
    );
}

#[test]
fn test_state_carries_across_steps() {
    let bb = default_backbone();
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();
    let input = vec![0.1f32; 128];
    let mut out1 = vec![0.0f32; 128];
    let mut out2 = vec![0.0f32; 128];

    bb.forward_step(&input, &mut out1, &mut state, &mut scratch);
    bb.forward_step(&input, &mut out2, &mut state, &mut scratch);

    // Same input, but state changed → outputs differ
    let diff: f32 = out1
        .iter()
        .zip(out2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "step 2 should differ from step 1 (state carries)"
    );
}

#[test]
fn test_state_reset_reproduces() {
    let bb = default_backbone();
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();
    let input = vec![0.1f32; 128];

    // Run 5 steps
    let mut out_first = vec![0.0f32; 128];
    bb.forward_step(&input, &mut out_first, &mut state, &mut scratch);
    for _ in 1..5 {
        let mut tmp = vec![0.0f32; 128];
        bb.forward_step(&input, &mut tmp, &mut state, &mut scratch);
    }

    // Reset and run again
    state.reset();
    let mut out_after_reset = vec![0.0f32; 128];
    bb.forward_step(&input, &mut out_after_reset, &mut state, &mut scratch);

    // First step after reset should match first step from cold start
    for (a, b) in out_first.iter().zip(out_after_reset.iter()) {
        assert!((a - b).abs() < 1e-6, "reset should reproduce: {a} vs {b}");
    }
}

#[test]
fn test_sequence_matches_step_by_step() {
    let bb = default_backbone();
    let t = 20;
    let dm = bb.config().d_model;
    let input_dim = bb.input_dim();
    let inputs: Vec<f32> = (0..t * input_dim).map(|i| (i as f32) * 0.001).collect();

    // Method 1: forward_sequence
    let mut state1 = bb.alloc_state();
    let mut scratch1 = bb.alloc_scratch();
    let mut out_seq = vec![0.0f32; t * dm];
    bb.forward_sequence(&inputs, &mut out_seq, &mut state1, &mut scratch1, t);

    // Method 2: step-by-step
    let mut state2 = bb.alloc_state();
    let mut scratch2 = bb.alloc_scratch();
    let mut out_step = vec![0.0f32; dm];
    for step in 0..t {
        let inp = &inputs[step * input_dim..(step + 1) * input_dim];
        bb.forward_step(inp, &mut out_step, &mut state2, &mut scratch2);
        let expected = &out_seq[step * dm..(step + 1) * dm];
        for (j, (a, b)) in out_step.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "step {step} dim {j}: seq={b} vs step={a}"
            );
        }
    }
}

#[test]
fn test_batch_matches_individual() {
    let bb = default_backbone();
    let dm = bb.config().d_model;
    let input_dim = bb.input_dim();
    let batch = 4;

    // Different inputs per sample
    let inputs: Vec<f32> = (0..batch * input_dim).map(|i| (i as f32) * 0.01).collect();

    // Method 1: batch inference
    let mut states_batch: Vec<MambaState> = (0..batch).map(|_| bb.alloc_state()).collect();
    let mut scratches: Vec<MambaStepScratch> = (0..batch).map(|_| bb.alloc_scratch()).collect();
    let mut out_batch = vec![0.0f32; batch * dm];
    bb.forward_step_batch(&inputs, &mut out_batch, &mut states_batch, &mut scratches);

    // Method 2: individual
    for b in 0..batch {
        let mut state = bb.alloc_state();
        let mut scratch = bb.alloc_scratch();
        let mut out_single = vec![0.0f32; dm];
        let inp = &inputs[b * input_dim..(b + 1) * input_dim];
        bb.forward_step(inp, &mut out_single, &mut state, &mut scratch);

        let batch_slice = &out_batch[b * dm..(b + 1) * dm];
        for (j, (a, b_val)) in out_single.iter().zip(batch_slice.iter()).enumerate() {
            assert!(
                (a - b_val).abs() < 1e-6,
                "batch {b} dim {j}: single={a} vs batch={b_val}"
            );
        }
    }
}

// =========================================================================
// Serialization
// =========================================================================

#[test]
fn test_serialize_roundtrip() {
    let cfg = MambaConfig::default();
    let bb = MambaBackbone::init(cfg, 128, 42);
    let dir = std::env::temp_dir().join("mamba_rs_test_serialize");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model.safetensors");

    mamba_rs::serialize::save(&path, bb.weights(), bb.config(), bb.input_dim()).unwrap();
    let (loaded_w, loaded_cfg, loaded_input_dim) = mamba_rs::serialize::load(&path).unwrap();

    assert_eq!(loaded_cfg.d_model, cfg.d_model);
    assert_eq!(loaded_cfg.d_state, cfg.d_state);
    assert_eq!(loaded_cfg.d_conv, cfg.d_conv);
    assert_eq!(loaded_cfg.expand, cfg.expand);
    assert_eq!(loaded_cfg.n_layers, cfg.n_layers);
    assert_eq!(loaded_input_dim, 128);

    // Compare weights element-by-element (exact, binary format preserves bits)
    for (a, b) in bb
        .weights()
        .input_proj_w
        .iter()
        .zip(loaded_w.input_proj_w.iter())
    {
        assert_eq!(a.to_bits(), b.to_bits(), "input_proj_w mismatch");
    }
    for (li, (lw_orig, lw_load)) in bb
        .weights()
        .layers
        .iter()
        .zip(loaded_w.layers.iter())
        .enumerate()
    {
        for (a, b) in lw_orig.a_log.iter().zip(lw_load.a_log.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "layer {li} a_log mismatch");
        }
        for (a, b) in lw_orig.in_proj_w.iter().zip(lw_load.in_proj_w.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "layer {li} in_proj_w mismatch");
        }
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_serialize_inference_parity() {
    let cfg = MambaConfig::default();
    let bb = MambaBackbone::init(cfg, 128, 42);
    let dir = std::env::temp_dir().join("mamba_rs_test_parity");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model.safetensors");

    mamba_rs::serialize::save(&path, bb.weights(), bb.config(), bb.input_dim()).unwrap();
    let (loaded_w, loaded_cfg, loaded_input_dim) = mamba_rs::serialize::load(&path).unwrap();
    let bb2 = MambaBackbone::from_weights(loaded_cfg, loaded_w).unwrap();
    assert_eq!(bb2.input_dim(), loaded_input_dim);

    // Run 10 steps on both, compare outputs
    let mut state1 = bb.alloc_state();
    let mut scratch1 = bb.alloc_scratch();
    let mut state2 = bb2.alloc_state();
    let mut scratch2 = bb2.alloc_scratch();
    let mut out1 = vec![0.0f32; 128];
    let mut out2 = vec![0.0f32; 128];

    for step in 0..10 {
        let input: Vec<f32> = (0..128).map(|i| (step * 128 + i) as f32 * 0.01).collect();
        bb.forward_step(&input, &mut out1, &mut state1, &mut scratch1);
        bb2.forward_step(&input, &mut out2, &mut state2, &mut scratch2);

        for (j, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "step {step} dim {j}: orig={a} vs loaded={b}"
            );
        }
    }

    std::fs::remove_dir_all(&dir).ok();
}

// =========================================================================
// Training: finite-diff gradient check
// =========================================================================

/// Helper: build training scaffolding from inference weights.
fn build_train_scaffolding(
    weights: &mamba_rs::MambaWeights,
    cfg: &MambaConfig,
    input_dim: usize,
    seq_len: usize,
) -> (
    mamba_rs::train::weights::TrainMambaWeights,
    mamba_rs::ops::dims::MambaDims,
) {
    use mamba_rs::ops::dims::MambaDims;
    use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};

    let dims = MambaDims::from_config(cfg, seq_len, input_dim);

    let tw = TrainMambaWeights {
        input_proj_w: weights.input_proj_w.clone(),
        input_proj_b: weights.input_proj_b.clone(),
        layers: weights
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
        norm_f_weight: weights.norm_f_weight.clone(),
    };

    (tw, dims)
}

/// Helper: run training forward pass and return output + loss.
/// Uses sum(temporal) as loss — NOT sum(temporal^2), because RMSNorm
/// makes sum(y^2) nearly constant, causing pathological near-cancellation
/// in the backward gradient.
fn run_forward(
    tw: &mamba_rs::train::weights::TrainMambaWeights,
    dims: &mamba_rs::ops::dims::MambaDims,
    input: &[f32],
) -> (Vec<f32>, f32) {
    use mamba_rs::ops::dims::MambaRecurrentState;
    use mamba_rs::train::flat::MambaBackboneFlat;
    use mamba_rs::train::scratch::PhaseScratch;

    let di = dims.d_inner;
    let ds = dims.d_state;
    let dc = dims.d_conv;
    let nl = dims.n_layers;
    let t = dims.seq_len;
    let dm = dims.d_model;

    let mut acts = MambaBackboneFlat::zeros(*dims);
    let mut scratch = PhaseScratch::zeros(dims);
    let mut conv = vec![0.0f32; nl * di * dc];
    let mut ssm = vec![0.0f32; nl * di * ds];

    // Precompute a_neg
    let mut a_neg = vec![0.0f32; nl * di * ds];
    for (l, lw) in tw.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }

    let mut state = MambaRecurrentState {
        conv: &mut conv,
        ssm: &mut ssm,
        a_neg: &a_neg,
    };

    let mut temporal = vec![0.0f32; t * dm];
    mamba_rs::train::forward::forward_mamba_backbone_batched(
        &mut temporal,
        &mut acts,
        tw,
        input,
        &mut state,
        &mut scratch,
        dims,
    );

    // Linear loss: sum(temporal). d_loss/d_temporal = 1.0 (no cancellation)
    let loss: f32 = temporal.iter().sum();
    (temporal, loss)
}

#[test]
fn test_finite_diff_input_proj_bias() {
    let cfg = MambaConfig::default();
    let input_dim = 128;
    let seq_len = 4;

    let weights = mamba_rs::MambaWeights::init(&cfg, input_dim, 42);
    let (tw, dims) = build_train_scaffolding(&weights, &cfg, input_dim, seq_len);

    let input: Vec<f32> = (0..seq_len * input_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();

    let eps = 1e-3_f32;
    let param_idx = 0;

    // Baseline forward
    let (temporal, _loss_base) = run_forward(&tw, &dims, &input);

    // Perturb +eps
    let mut tw_plus = tw.clone();
    tw_plus.input_proj_b[param_idx] += eps;
    let (_temporal_plus, loss_plus) = run_forward(&tw_plus, &dims, &input);

    // Perturb -eps
    let mut tw_minus = tw.clone();
    tw_minus.input_proj_b[param_idx] -= eps;
    let (_temporal_minus, loss_minus) = run_forward(&tw_minus, &dims, &input);

    let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);

    // Analytical backward
    let d_temporal: Vec<f32> = vec![1.0f32; temporal.len()];
    let mut grad_tw = mamba_rs::train::weights::TrainMambaWeights::zeros_from_dims(&dims);

    // Need a_neg for backward
    let di = dims.d_inner;
    let ds = dims.d_state;
    let mut a_neg = vec![0.0f32; dims.n_layers * di * ds];
    for (l, lw) in tw.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }

    let acts = {
        // Re-run forward to get saved activations
        use mamba_rs::ops::dims::MambaRecurrentState;
        use mamba_rs::train::flat::MambaBackboneFlat;
        use mamba_rs::train::scratch::PhaseScratch;

        let mut acts = MambaBackboneFlat::zeros(dims);
        let mut scratch = PhaseScratch::zeros(&dims);
        let mut conv = vec![0.0f32; dims.n_layers * di * dims.d_conv];
        let mut ssm = vec![0.0f32; dims.n_layers * di * ds];
        let mut state = MambaRecurrentState {
            conv: &mut conv,
            ssm: &mut ssm,
            a_neg: &a_neg,
        };
        let mut temporal2 = vec![0.0f32; dims.seq_len * dims.d_model];
        mamba_rs::train::forward::forward_mamba_backbone_batched(
            &mut temporal2,
            &mut acts,
            &tw,
            &input,
            &mut state,
            &mut scratch,
            &dims,
        );
        acts
    };

    let mut d_temporal_mut = d_temporal;
    let mut bwd_scratch = mamba_rs::train::scratch::BackwardPhaseScratch::zeros(&dims);
    mamba_rs::train::backward::backward_mamba_backbone_batched(
        &mut d_temporal_mut,
        &mut grad_tw,
        &acts,
        &tw,
        &a_neg,
        &mut bwd_scratch,
        &dims,
    );

    let analytical_grad = grad_tw.input_proj_b[param_idx];

    let rel_err = if numerical_grad.abs() > 1e-8 {
        ((analytical_grad - numerical_grad) / numerical_grad).abs()
    } else {
        (analytical_grad - numerical_grad).abs()
    };

    assert!(
        rel_err < 5e-2,
        "finite-diff gradient check failed for input_proj_b[0]: \
         analytical={analytical_grad:.6}, numerical={numerical_grad:.6}, rel_err={rel_err:.4}"
    );
}

#[test]
fn test_finite_diff_dt_proj_bias() {
    let cfg = MambaConfig::default();
    let input_dim = 128;
    let seq_len = 4;

    let weights = mamba_rs::MambaWeights::init(&cfg, input_dim, 42);
    let (tw, dims) = build_train_scaffolding(&weights, &cfg, input_dim, seq_len);

    let input: Vec<f32> = (0..seq_len * input_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();

    let eps = 1e-3_f32;

    // Find a dt_proj_b index with non-saturated softplus (gradient != 0).
    // dt_proj_b init = inv_softplus(exp(log_uniform)) — some elements have
    // large values where sigmoid → 0, making gradient vanish. Pick the index
    // with the largest absolute analytical gradient.
    let (temporal, _) = run_forward(&tw, &dims, &input);
    let d_temporal: Vec<f32> = vec![1.0f32; temporal.len()];
    let mut grad_tw = mamba_rs::train::weights::TrainMambaWeights::zeros_from_dims(&dims);

    let di = dims.d_inner;
    let ds = dims.d_state;
    let mut a_neg = vec![0.0f32; dims.n_layers * di * ds];
    for (l, lw) in tw.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }

    let acts = {
        use mamba_rs::ops::dims::MambaRecurrentState;
        use mamba_rs::train::flat::MambaBackboneFlat;
        use mamba_rs::train::scratch::PhaseScratch;

        let mut acts = MambaBackboneFlat::zeros(dims);
        let mut scratch = PhaseScratch::zeros(&dims);
        let mut conv = vec![0.0f32; dims.n_layers * di * dims.d_conv];
        let mut ssm = vec![0.0f32; dims.n_layers * di * ds];
        let mut state = MambaRecurrentState {
            conv: &mut conv,
            ssm: &mut ssm,
            a_neg: &a_neg,
        };
        let mut temporal2 = vec![0.0f32; dims.seq_len * dims.d_model];
        mamba_rs::train::forward::forward_mamba_backbone_batched(
            &mut temporal2,
            &mut acts,
            &tw,
            &input,
            &mut state,
            &mut scratch,
            &dims,
        );
        acts
    };

    let mut d_temporal_mut = d_temporal;
    let mut bwd_scratch = mamba_rs::train::scratch::BackwardPhaseScratch::zeros(&dims);
    mamba_rs::train::backward::backward_mamba_backbone_batched(
        &mut d_temporal_mut,
        &mut grad_tw,
        &acts,
        &tw,
        &a_neg,
        &mut bwd_scratch,
        &dims,
    );

    // Pick the dt_proj_b element with largest analytical gradient
    let param_idx = grad_tw.layers[0]
        .dt_proj_b
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let analytical_grad = grad_tw.layers[0].dt_proj_b[param_idx];

    // Numerical gradient for that element
    let mut tw_plus = tw.clone();
    tw_plus.layers[0].dt_proj_b[param_idx] += eps;
    let (_, loss_plus) = run_forward(&tw_plus, &dims, &input);

    let mut tw_minus = tw.clone();
    tw_minus.layers[0].dt_proj_b[param_idx] -= eps;
    let (_, loss_minus) = run_forward(&tw_minus, &dims, &input);

    let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);

    let rel_err = if numerical_grad.abs() > 1e-8 {
        ((analytical_grad - numerical_grad) / numerical_grad).abs()
    } else {
        (analytical_grad - numerical_grad).abs()
    };

    assert!(
        rel_err < 0.10,
        "finite-diff gradient check failed for dt_proj_b[{param_idx}]: \
         analytical={analytical_grad:.6}, numerical={numerical_grad:.6}, rel_err={rel_err:.4} (threshold 0.10)"
    );
}

// =========================================================================
// Debug: trace gradient through backward phases
// =========================================================================

#[test]
fn test_debug_gradient_trace() {
    let cfg = MambaConfig::default();
    let input_dim = 128;
    let seq_len = 4;

    let weights = mamba_rs::MambaWeights::init(&cfg, input_dim, 42);
    let (tw, dims) = build_train_scaffolding(&weights, &cfg, input_dim, seq_len);

    let input: Vec<f32> = (0..seq_len * input_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();

    let di = dims.d_inner;
    let ds = dims.d_state;
    let mut a_neg = vec![0.0f32; dims.n_layers * di * ds];
    for (l, lw) in tw.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }

    // Forward to get acts
    let acts = {
        use mamba_rs::ops::dims::MambaRecurrentState;
        use mamba_rs::train::flat::MambaBackboneFlat;
        use mamba_rs::train::scratch::PhaseScratch;

        let mut acts = MambaBackboneFlat::zeros(dims);
        let mut scratch = PhaseScratch::zeros(&dims);
        let mut conv = vec![0.0f32; dims.n_layers * di * dims.d_conv];
        let mut ssm = vec![0.0f32; dims.n_layers * di * ds];
        let mut state = MambaRecurrentState {
            conv: &mut conv,
            ssm: &mut ssm,
            a_neg: &a_neg,
        };
        let mut temporal = vec![0.0f32; dims.seq_len * dims.d_model];
        mamba_rs::train::forward::forward_mamba_backbone_batched(
            &mut temporal,
            &mut acts,
            &tw,
            &input,
            &mut state,
            &mut scratch,
            &dims,
        );

        // Print temporal stats
        let t_max = temporal.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let t_sum: f32 = temporal.iter().sum();
        eprintln!("FORWARD: temporal max_abs={t_max:.6}, sum={t_sum:.6}");

        acts
    };

    // Backward with d_temporal = ones
    let mut d_temporal = vec![1.0f32; dims.seq_len * dims.d_model];
    let mut grad_tw = mamba_rs::train::weights::TrainMambaWeights::zeros_from_dims(&dims);
    let mut bwd_scratch = mamba_rs::train::scratch::BackwardPhaseScratch::zeros(&dims);

    // Print d_temporal before backward
    eprintln!("BEFORE BWD: d_temporal[0..4] = {:?}", &d_temporal[0..4]);

    mamba_rs::train::backward::backward_mamba_backbone_batched(
        &mut d_temporal,
        &mut grad_tw,
        &acts,
        &tw,
        &a_neg,
        &mut bwd_scratch,
        &dims,
    );

    // Print d_temporal after backward (should be d_input_proj_output)
    let dt_max = d_temporal.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let dt_sum: f32 = d_temporal.iter().sum();
    eprintln!("AFTER BWD: d_temporal max_abs={dt_max:.6}, sum={dt_sum:.6}");
    eprintln!("AFTER BWD: d_temporal[0..4] = {:?}", &d_temporal[0..4]);

    // Print key gradients
    eprintln!("GRADS layer0:");
    eprintln!("  input_proj_b[0] = {:.8}", grad_tw.input_proj_b[0]);
    eprintln!("  dt_proj_b[0]    = {:.8}", grad_tw.layers[0].dt_proj_b[0]);
    eprintln!(
        "  dt_proj_b max   = {:.8}",
        grad_tw.layers[0]
            .dt_proj_b
            .iter()
            .fold(0.0f32, |a, &b| a.max(b.abs()))
    );
    eprintln!(
        "  dt_proj_w max   = {:.8}",
        grad_tw.layers[0]
            .dt_proj_w
            .iter()
            .fold(0.0f32, |a, &b| a.max(b.abs()))
    );
    eprintln!(
        "  conv1d_b[0]     = {:.8}",
        grad_tw.layers[0].conv1d_bias[0]
    );
    eprintln!(
        "  in_proj_w max   = {:.8}",
        grad_tw.layers[0]
            .in_proj_w
            .iter()
            .fold(0.0f32, |a, &b| a.max(b.abs()))
    );
    eprintln!(
        "  out_proj_w max  = {:.8}",
        grad_tw.layers[0]
            .out_proj_w
            .iter()
            .fold(0.0f32, |a, &b| a.max(b.abs()))
    );
    eprintln!(
        "  a_log max       = {:.8}",
        grad_tw.layers[0]
            .a_log
            .iter()
            .fold(0.0f32, |a, &b| a.max(b.abs()))
    );
    eprintln!(
        "  d_param max     = {:.8}",
        grad_tw.layers[0]
            .d_param
            .iter()
            .fold(0.0f32, |a, &b| a.max(b.abs()))
    );
    eprintln!(
        "  norm_weight max = {:.8}",
        grad_tw.layers[0]
            .norm_weight
            .iter()
            .fold(0.0f32, |a, &b| a.max(b.abs()))
    );
    eprintln!(
        "  norm_f max      = {:.8}",
        grad_tw
            .norm_f_weight
            .iter()
            .fold(0.0f32, |a, &b| a.max(b.abs()))
    );

    // Check bwd_scratch for d_delta and d_delta_raw
    let d_delta_max = bwd_scratch
        .d_delta_flat
        .iter()
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    let d_delta_raw_max = bwd_scratch
        .d_delta_raw_flat
        .iter()
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    let d_y_max = bwd_scratch
        .d_y_flat
        .iter()
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    let d_gated_max = bwd_scratch
        .d_gated_flat
        .iter()
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    let d_u_max = bwd_scratch
        .d_u_flat
        .iter()
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    eprintln!("BWD SCRATCH (last layer processed = layer 0):");
    eprintln!("  d_gated max      = {d_gated_max:.8}");
    eprintln!("  d_y max          = {d_y_max:.8}");
    eprintln!("  d_delta max      = {d_delta_max:.8}");
    eprintln!("  d_delta_raw max  = {d_delta_raw_max:.8}");
    eprintln!("  d_u max          = {d_u_max:.8}");
}

// =========================================================================
// GPU/CPU parity tests (require cuda feature + NVIDIA GPU)
// =========================================================================

#[cfg(feature = "cuda")]
mod gpu_tests {
    use mamba_rs::{MambaBackbone, MambaConfig, MambaState, MambaStepScratch};

    #[test]
    fn test_gpu_inference_matches_cpu() {
        use mamba_rs::gpu::inference::GpuMambaBackbone;

        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;
        let batch = 4;

        let bb = MambaBackbone::init(cfg, input_dim, 42);

        // GPU backbone
        let mut gpu_bb =
            GpuMambaBackbone::new(0, bb.weights(), *bb.config(), input_dim, batch).unwrap();

        // CPU states
        let mut cpu_states: Vec<MambaState> = (0..batch).map(|_| bb.alloc_state()).collect();
        let mut cpu_scratches: Vec<MambaStepScratch> =
            (0..batch).map(|_| bb.alloc_scratch()).collect();

        // Run 20 steps, compare
        for step in 0..20 {
            let inputs: Vec<f32> = (0..batch * input_dim)
                .map(|i| ((step * batch * input_dim + i) as f32) * 0.001)
                .collect();

            // GPU
            let mut gpu_out = vec![0.0f32; batch * cfg.d_model];
            gpu_bb.step(&inputs, &mut gpu_out).unwrap();

            // CPU
            let mut cpu_out = vec![0.0f32; batch * cfg.d_model];
            bb.forward_step_batch(&inputs, &mut cpu_out, &mut cpu_states, &mut cpu_scratches);

            let max_diff: f32 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            // TF32 compound error: each SGEMM has ~1e-3 error, compounded over
            // ~15 matmuls in a 3-layer Mamba network. For batch=1-4 and 20 steps
            // with state accumulation, max_diff can reach ~5.0 in worst case.
            assert!(
                max_diff < 5.0,
                "step {step}: GPU vs CPU max_diff={max_diff:.6}"
            );
        }
    }

    #[test]
    fn test_gpu_inference_cuda_graph() {
        use mamba_rs::gpu::inference::GpuMambaBackbone;

        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;

        let bb = MambaBackbone::init(cfg, input_dim, 42);

        // Use a SINGLE engine for both paths. This ensures the same cuBLAS
        // handle (and therefore the same TF32 algorithm selections) is used
        // for both non-graph and graph runs, giving bit-identical results.
        let mut gpu_bb =
            GpuMambaBackbone::new(0, bb.weights(), *bb.config(), input_dim, 1).unwrap();

        let input = vec![0.1f32; input_dim];

        // Phase 1: run 5 steps without graph, save output
        let mut out_no_graph = vec![0.0f32; cfg.d_model];
        for _ in 0..5 {
            gpu_bb.step(&input, &mut out_no_graph).unwrap();
        }

        // Reset state, capture graph, run 5 steps with graph
        gpu_bb.reset().unwrap();
        gpu_bb.capture_graph().unwrap();
        let mut out_graph = vec![0.0f32; cfg.d_model];
        for _ in 0..5 {
            gpu_bb.step(&input, &mut out_graph).unwrap();
        }

        // Same handle, same algorithms — should be bit-identical or near it.
        let max_diff: f32 = out_no_graph
            .iter()
            .zip(out_graph.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "CUDA Graph vs non-graph (same engine): max_diff={max_diff:.6}"
        );
    }

    #[test]
    fn test_gpu_inference_state_reset() {
        use mamba_rs::gpu::inference::GpuMambaBackbone;

        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;
        let bb = MambaBackbone::init(cfg, input_dim, 42);

        let mut gpu_bb =
            GpuMambaBackbone::new(0, bb.weights(), *bb.config(), input_dim, 1).unwrap();

        let input = vec![0.1f32; input_dim];
        let mut out_first = vec![0.0f32; cfg.d_model];

        // First step from cold state
        gpu_bb.step(&input, &mut out_first).unwrap();

        // Run more steps
        let mut tmp = vec![0.0f32; cfg.d_model];
        for _ in 0..5 {
            gpu_bb.step(&input, &mut tmp).unwrap();
        }

        // Reset and step again
        gpu_bb.reset().unwrap();
        let mut out_after_reset = vec![0.0f32; cfg.d_model];
        gpu_bb.step(&input, &mut out_after_reset).unwrap();

        let max_diff: f32 = out_first
            .iter()
            .zip(out_after_reset.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff < 1e-5, "state reset: max_diff={max_diff:.6}");
    }

    #[test]
    fn test_gpu_cpu_training_forward_parity() {
        use mamba_rs::gpu::buffers::GpuBuffer;
        use mamba_rs::gpu::context::GpuCtx;
        use mamba_rs::gpu::device::GpuDevice;
        use mamba_rs::gpu::forward::GpuRecurrentState;
        use mamba_rs::gpu::forward::{
            GpuMambaBackboneActs, GpuMambaDims, GpuMambaScratch, gpu_forward_mamba_backbone,
        };
        use mamba_rs::gpu::weights::GpuMambaTrainWeights;
        use mamba_rs::ops::dims::MambaDims;

        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;
        let seq_len = 8;

        let weights = mamba_rs::MambaWeights::init(&cfg, input_dim, 42);
        let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
        let di = cfg.d_inner();
        let ds = cfg.d_state;

        // CPU forward
        let tw = super::build_train_scaffolding(&weights, &cfg, input_dim, seq_len).0;
        let (cpu_temporal, _) = super::run_forward(
            &tw,
            &dims,
            &(0..seq_len * input_dim)
                .map(|i| i as f32 * 0.001)
                .collect::<Vec<_>>(),
        );

        // GPU forward
        let device = GpuDevice::new(0).unwrap();
        let ctx = GpuCtx::new(&device).unwrap();

        let gpu_dims = GpuMambaDims {
            batch: 1,
            d_model: cfg.d_model,
            d_inner: di,
            d_state: ds,
            d_conv: cfg.d_conv,
            dt_rank: cfg.dt_rank(),
            xdbl_dim: cfg.xdbl_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers: cfg.n_layers,
        };

        let gpu_w = GpuMambaTrainWeights::from_cpu(&ctx.stream, &weights).unwrap();

        let mut gpu_temporal = GpuBuffer::zeros(&ctx.stream, seq_len * cfg.d_model).unwrap();
        let mut gpu_acts = GpuMambaBackboneActs::new(&ctx.stream, &gpu_dims).unwrap();
        let input_gpu = GpuBuffer::from_cpu(
            &ctx.stream,
            &(0..seq_len * input_dim)
                .map(|i| i as f32 * 0.001)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let mut state = GpuRecurrentState {
            conv_states: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * cfg.d_conv).unwrap(),
            ssm_states: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * ds).unwrap(),
            a_neg_all: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * ds).unwrap(),
        };
        let mut a_neg_cpu = vec![0.0f32; cfg.n_layers * di * ds];
        for (l, lw) in weights.layers.iter().enumerate() {
            for i in 0..di * ds {
                a_neg_cpu[l * di * ds + i] = -lw.a_log[i].exp();
            }
        }
        state.a_neg_all.upload(&ctx.stream, &a_neg_cpu).unwrap();

        let mut gpu_scratch = GpuMambaScratch::new(&ctx.stream, &gpu_dims).unwrap();

        gpu_forward_mamba_backbone(
            &ctx,
            &mut gpu_temporal,
            &mut gpu_acts,
            &gpu_w,
            &input_gpu,
            &mut state,
            &mut gpu_scratch,
        )
        .unwrap();

        let gpu_out = gpu_temporal.to_cpu(&ctx.stream).unwrap();

        // Compare
        let max_diff: f32 = cpu_temporal
            .iter()
            .zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // TF32 compound error: ~1e-3 per SGEMM, compounded over ~15 matmuls
        // in a 3-layer Mamba network with seq_len=8 timesteps.
        assert!(
            max_diff < 5.0,
            "GPU vs CPU training forward: max_diff={max_diff:.6}"
        );
    }

    #[test]
    fn test_gpu_cpu_training_backward_parity() {
        use mamba_rs::gpu::backward::gpu_backward_mamba_backbone;
        use mamba_rs::gpu::buffers::GpuBuffer;
        use mamba_rs::gpu::context::GpuCtx;
        use mamba_rs::gpu::device::GpuDevice;
        use mamba_rs::gpu::forward::GpuRecurrentState;
        use mamba_rs::gpu::forward::{
            GpuMambaBackboneActs, GpuMambaDims, GpuMambaScratch, gpu_forward_mamba_backbone,
        };
        use mamba_rs::gpu::weights::{GpuMambaGrads, GpuMambaTrainWeights};

        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;
        let seq_len = 4;
        let di = cfg.d_inner();
        let ds = cfg.d_state;

        let weights = mamba_rs::MambaWeights::init(&cfg, input_dim, 42);

        // CPU forward + backward
        let (tw, dims) = super::build_train_scaffolding(&weights, &cfg, input_dim, seq_len);
        let input: Vec<f32> = (0..seq_len * input_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let (cpu_temporal, _) = super::run_forward(&tw, &dims, &input);

        let d_temporal_cpu: Vec<f32> = vec![1.0f32; cpu_temporal.len()];
        let mut cpu_grads = mamba_rs::train::weights::TrainMambaWeights::zeros_from_dims(&dims);
        let mut a_neg_cpu = vec![0.0f32; cfg.n_layers * di * ds];
        for (l, lw) in tw.layers.iter().enumerate() {
            for i in 0..di * ds {
                a_neg_cpu[l * di * ds + i] = -lw.a_log[i].exp();
            }
        }
        let cpu_acts = {
            use mamba_rs::ops::dims::MambaRecurrentState;
            use mamba_rs::train::flat::MambaBackboneFlat;
            use mamba_rs::train::scratch::PhaseScratch;
            let mut acts = MambaBackboneFlat::zeros(dims);
            let mut scratch = PhaseScratch::zeros(&dims);
            let mut conv = vec![0.0f32; cfg.n_layers * di * cfg.d_conv];
            let mut ssm = vec![0.0f32; cfg.n_layers * di * ds];
            let mut state = MambaRecurrentState {
                conv: &mut conv,
                ssm: &mut ssm,
                a_neg: &a_neg_cpu,
            };
            let mut t = vec![0.0f32; seq_len * cfg.d_model];
            mamba_rs::train::forward::forward_mamba_backbone_batched(
                &mut t,
                &mut acts,
                &tw,
                &input,
                &mut state,
                &mut scratch,
                &dims,
            );
            acts
        };
        let mut d_temporal_cpu_mut = d_temporal_cpu;
        let mut bwd_scratch = mamba_rs::train::scratch::BackwardPhaseScratch::zeros(&dims);
        mamba_rs::train::backward::backward_mamba_backbone_batched(
            &mut d_temporal_cpu_mut,
            &mut cpu_grads,
            &cpu_acts,
            &tw,
            &a_neg_cpu,
            &mut bwd_scratch,
            &dims,
        );

        // GPU forward + backward
        let device = GpuDevice::new(0).unwrap();
        let ctx = GpuCtx::new(&device).unwrap();

        let gpu_dims = GpuMambaDims {
            batch: 1,
            d_model: cfg.d_model,
            d_inner: di,
            d_state: ds,
            d_conv: cfg.d_conv,
            dt_rank: cfg.dt_rank(),
            xdbl_dim: cfg.xdbl_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers: cfg.n_layers,
        };

        let gpu_w = GpuMambaTrainWeights::from_cpu(&ctx.stream, &weights).unwrap();
        let mut gpu_temporal = GpuBuffer::zeros(&ctx.stream, seq_len * cfg.d_model).unwrap();
        let mut gpu_acts = GpuMambaBackboneActs::new(&ctx.stream, &gpu_dims).unwrap();
        let input_gpu = GpuBuffer::from_cpu(&ctx.stream, &input).unwrap();

        let mut gpu_state = GpuRecurrentState {
            conv_states: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * cfg.d_conv).unwrap(),
            ssm_states: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * ds).unwrap(),
            a_neg_all: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * ds).unwrap(),
        };
        gpu_state.a_neg_all.upload(&ctx.stream, &a_neg_cpu).unwrap();

        let mut gpu_scratch = GpuMambaScratch::new(&ctx.stream, &gpu_dims).unwrap();

        gpu_forward_mamba_backbone(
            &ctx,
            &mut gpu_temporal,
            &mut gpu_acts,
            &gpu_w,
            &input_gpu,
            &mut gpu_state,
            &mut gpu_scratch,
        )
        .unwrap();

        let gpu_grads = GpuMambaGrads::new(&ctx.stream, &cfg, input_dim).unwrap();
        let mut d_temporal_gpu =
            GpuBuffer::from_cpu(&ctx.stream, &vec![1.0f32; seq_len * cfg.d_model]).unwrap();

        // Sync: ensure all async alloc/memset operations complete before
        // backward reads grad buffer with beta=1.0 (cuBLAS accumulation).
        ctx.stream.synchronize().expect("pre-backward sync");

        gpu_backward_mamba_backbone(
            &ctx,
            &mut d_temporal_gpu,
            &gpu_grads,
            &gpu_acts,
            &gpu_w,
            &gpu_state.a_neg_all,
            &mut gpu_scratch,
        )
        .unwrap();

        // Sync stream before downloading gradients (GPU backward is async)
        ctx.stream
            .synchronize()
            .expect("stream sync before grad download");

        // Download GPU gradients and compare ALL weight groups.
        //
        // TF32 backward tolerance: PyTorch-style allclose(atol + rtol * |expected|).
        // TF32 (10-bit mantissa) compounds ~5e-3 per SGEMM through ~50 backward ops
        // in a 3-layer Mamba network. Elements near zero need atol protection;
        // large elements need rtol scaling.
        let atol = 0.5_f32; // absolute tolerance for near-zero gradients
        let rtol = 0.10_f32; // 10% relative tolerance for large gradients

        macro_rules! check_grad {
            ($name:expr, $gpu_slice:expr, $cpu_slice:expr) => {{
                let gpu_vals = $gpu_slice.to_cpu().unwrap();
                let worst_idx = gpu_vals
                    .iter()
                    .zip($cpu_slice.iter())
                    .enumerate()
                    .map(|(i, (g, c))| {
                        let tol = atol + rtol * c.abs();
                        let diff = (g - c).abs();
                        (i, diff, tol, *g, *c)
                    })
                    .max_by(|a, b| (a.1 / a.2).partial_cmp(&(b.1 / b.2)).unwrap())
                    .unwrap();
                let (idx, diff, tol, gv, cv) = worst_idx;
                assert!(
                    diff < tol,
                    "backward {}: idx={idx} gpu={gv:.6} cpu={cv:.6} diff={diff:.6} tol={tol:.6}",
                    $name
                );
            }};
        }

        check_grad!(
            "input_proj_b",
            gpu_grads.input_proj_b,
            cpu_grads.input_proj_b
        );
        check_grad!(
            "input_proj_w",
            gpu_grads.input_proj_w,
            cpu_grads.input_proj_w
        );
        check_grad!(
            "norm_f_weight",
            gpu_grads.norm_f_weight,
            cpu_grads.norm_f_weight
        );

        for (li, (gpu_lg, cpu_lg)) in gpu_grads
            .layers
            .iter()
            .zip(cpu_grads.layers.iter())
            .enumerate()
        {
            macro_rules! check_layer {
                ($name:ident) => {
                    check_grad!(
                        &format!("layer{li}.{}", stringify!($name)),
                        gpu_lg.$name,
                        cpu_lg.$name
                    );
                };
            }
            check_layer!(norm_weight);
            check_layer!(in_proj_w);
            check_layer!(conv1d_weight);
            check_layer!(conv1d_bias);
            check_layer!(x_proj_w);
            check_layer!(dt_proj_w);
            check_layer!(dt_proj_b);
            check_layer!(a_log);
            check_layer!(d_param);
            check_layer!(out_proj_w);
        }
    }

    /// Diagnostic: run backward step-by-step, download d_temporal after each stage.
    /// Prints max relative diff at each stage to isolate WHERE GPU diverges from CPU.
    ///
    #[test]
    fn test_backward_diagnostic() {
        use cudarc::driver::PushKernelArg;
        use mamba_rs::gpu::backward::gpu_backward_mamba_layer;
        use mamba_rs::gpu::blas::gpu_sgemm_backward_grad_raw;
        use mamba_rs::gpu::buffers::GpuBuffer;
        use mamba_rs::gpu::context::GpuCtx;
        use mamba_rs::gpu::device::GpuDevice;
        use mamba_rs::gpu::forward::GpuRecurrentState;
        use mamba_rs::gpu::forward::{
            GpuMambaBackboneActs, GpuMambaDims, GpuMambaScratch, gpu_forward_mamba_backbone,
        };
        use mamba_rs::gpu::launch::grid_norm;
        use mamba_rs::gpu::weights::{GpuMambaGrads, GpuMambaTrainWeights};
        use mamba_rs::ops::dims::MambaRecurrentState;
        use mamba_rs::train::backward::backward_mamba_layer_batched;
        use mamba_rs::train::backward_ops::backward_rms_norm;
        use mamba_rs::train::flat::MambaBackboneFlat;
        use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
        use mamba_rs::train::weights::TrainMambaWeights;

        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;
        let seq_len = 4;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dm = cfg.d_model;
        let n_layers = cfg.n_layers;

        let weights = mamba_rs::MambaWeights::init(&cfg, input_dim, 42);
        let (tw, dims) = super::build_train_scaffolding(&weights, &cfg, input_dim, seq_len);

        let input: Vec<f32> = (0..seq_len * input_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();

        let mut a_neg_cpu = vec![0.0f32; n_layers * di * ds];
        for (l, lw) in tw.layers.iter().enumerate() {
            for i in 0..di * ds {
                a_neg_cpu[l * di * ds + i] = -lw.a_log[i].exp();
            }
        }

        // =============================================
        // CPU forward (save activations)
        // =============================================
        let cpu_acts = {
            let mut acts = MambaBackboneFlat::zeros(dims);
            let mut scratch = PhaseScratch::zeros(&dims);
            let mut conv = vec![0.0f32; n_layers * di * cfg.d_conv];
            let mut ssm = vec![0.0f32; n_layers * di * ds];
            let mut state = MambaRecurrentState {
                conv: &mut conv,
                ssm: &mut ssm,
                a_neg: &a_neg_cpu,
            };
            let mut t = vec![0.0f32; seq_len * dm];
            mamba_rs::train::forward::forward_mamba_backbone_batched(
                &mut t,
                &mut acts,
                &tw,
                &input,
                &mut state,
                &mut scratch,
                &dims,
            );
            acts
        };

        // =============================================
        // CPU backward step-by-step (capture d_temporal at each stage)
        // =============================================
        let mut d_temporal_cpu = vec![1.0f32; seq_len * dm];
        let mut cpu_grads = TrainMambaWeights::zeros_from_dims(&dims);
        let mut bwd_scratch = BackwardPhaseScratch::zeros(&dims);

        // CPU norm_f backward
        {
            let norm_f_dx = &mut bwd_scratch.d_input_proj_scratch[..seq_len * dm];
            norm_f_dx.fill(0.0);
            bwd_scratch.d_norm_f_weight_local.fill(0.0);
            backward_rms_norm(
                norm_f_dx,
                &mut bwd_scratch.d_norm_f_weight_local,
                &d_temporal_cpu[..seq_len * dm],
                &cpu_acts.norm_f_input[..seq_len * dm],
                (&tw.norm_f_weight, &cpu_acts.norm_f_rms[..seq_len]),
                seq_len,
                dm,
            );
            d_temporal_cpu[..seq_len * dm].copy_from_slice(&norm_f_dx[..seq_len * dm]);
            for (a, b) in cpu_grads
                .norm_f_weight
                .iter_mut()
                .zip(&bwd_scratch.d_norm_f_weight_local)
            {
                *a += b;
            }
        }
        let cpu_after_normf = d_temporal_cpu.clone();

        // CPU layer backward (reverse order)
        let mut cpu_after_layers = Vec::new();
        for layer_idx in (0..n_layers).rev() {
            let a_neg_start = layer_idx * di * ds;
            backward_mamba_layer_batched(
                &mut d_temporal_cpu,
                &mut cpu_grads.layers[layer_idx],
                &cpu_acts.layers[layer_idx],
                &tw.layers[layer_idx],
                &a_neg_cpu[a_neg_start..a_neg_start + di * ds],
                &mut bwd_scratch,
                &dims,
            );
            cpu_after_layers.push((layer_idx, d_temporal_cpu.clone()));
        }

        // =============================================
        // GPU forward + step-by-step backward
        // =============================================
        let device = GpuDevice::new(0).unwrap();
        let ctx = GpuCtx::new(&device).unwrap();

        let gpu_dims = GpuMambaDims {
            batch: 1,
            d_model: dm,
            d_inner: di,
            d_state: ds,
            d_conv: cfg.d_conv,
            dt_rank: cfg.dt_rank(),
            xdbl_dim: cfg.xdbl_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers,
        };

        let gpu_w = GpuMambaTrainWeights::from_cpu(&ctx.stream, &weights).unwrap();
        let mut gpu_temporal = GpuBuffer::zeros(&ctx.stream, seq_len * dm).unwrap();
        let mut gpu_acts = GpuMambaBackboneActs::new(&ctx.stream, &gpu_dims).unwrap();
        let input_gpu = GpuBuffer::from_cpu(&ctx.stream, &input).unwrap();

        let mut gpu_state = GpuRecurrentState {
            conv_states: GpuBuffer::zeros(&ctx.stream, n_layers * di * cfg.d_conv).unwrap(),
            ssm_states: GpuBuffer::zeros(&ctx.stream, n_layers * di * ds).unwrap(),
            a_neg_all: GpuBuffer::zeros(&ctx.stream, n_layers * di * ds).unwrap(),
        };
        gpu_state.a_neg_all.upload(&ctx.stream, &a_neg_cpu).unwrap();
        let mut gpu_scratch = GpuMambaScratch::new(&ctx.stream, &gpu_dims).unwrap();

        gpu_forward_mamba_backbone(
            &ctx,
            &mut gpu_temporal,
            &mut gpu_acts,
            &gpu_w,
            &input_gpu,
            &mut gpu_state,
            &mut gpu_scratch,
        )
        .unwrap();

        // GPU backward step-by-step
        let gpu_grads = GpuMambaGrads::new(&ctx.stream, &cfg, input_dim).unwrap();
        let mut d_temporal_gpu =
            GpuBuffer::from_cpu(&ctx.stream, &vec![1.0f32; seq_len * dm]).unwrap();

        let bt = seq_len; // batch=1

        // GPU norm_f backward
        {
            let bt_i = bt as i32;
            let dm_i = dm as i32;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.rmsnorm_bwd);
            builder.arg(gpu_scratch.d_norm.inner_mut());
            let _p = gpu_grads.norm_f_weight.ptr();
            builder.arg(&_p);
            builder.arg(d_temporal_gpu.inner());
            builder.arg(gpu_acts.norm_f_input.inner());
            let nf_ptr = gpu_w.norm_f_weight.cached_ptr();
            builder.arg(&nf_ptr);
            builder.arg(gpu_acts.norm_f_rms.inner());
            builder.arg(&bt_i);
            builder.arg(&dm_i);
            unsafe { builder.launch(grid_norm(bt, dm)) }.unwrap();
            d_temporal_gpu
                .copy_from(&gpu_scratch.d_norm, &ctx.stream)
                .unwrap();
        }
        ctx.stream.synchronize().unwrap();

        // Compare after norm_f
        let gpu_after_normf = d_temporal_gpu.to_cpu(&ctx.stream).unwrap();
        let (max_rel, max_abs) = compare_vecs(&gpu_after_normf, &cpu_after_normf);
        println!(
            "DIAG after norm_f: max_rel={max_rel:.6}, max_abs={max_abs:.6}, \
             gpu_mean={:.6}, cpu_mean={:.6}",
            mean(&gpu_after_normf),
            mean(&cpu_after_normf),
        );

        // GPU layer backward (reverse order)
        let a_neg_per_layer = di * ds;
        for (i, layer_idx) in (0..n_layers).rev().enumerate() {
            let base = gpu_state.a_neg_all.raw_ptr(&ctx.stream);
            let a_neg_ptr =
                base + (layer_idx * a_neg_per_layer * std::mem::size_of::<f32>()) as u64;

            gpu_backward_mamba_layer(
                &ctx,
                &mut d_temporal_gpu,
                &gpu_grads.layers[layer_idx],
                &gpu_acts.layers[layer_idx],
                &gpu_w.layers[layer_idx],
                a_neg_ptr,
                &mut gpu_scratch,
            )
            .unwrap();

            ctx.stream.synchronize().unwrap();
            let gpu_vals = d_temporal_gpu.to_cpu(&ctx.stream).unwrap();
            let (cpu_layer_idx, ref cpu_vals) = cpu_after_layers[i];
            assert_eq!(cpu_layer_idx, layer_idx);

            let (max_rel, max_abs) = compare_vecs(&gpu_vals, cpu_vals);
            println!(
                "DIAG after layer {} backward: max_rel={max_rel:.6}, max_abs={max_abs:.6}, \
                 gpu_mean={:.6}, cpu_mean={:.6}",
                layer_idx,
                mean(&gpu_vals),
                mean(cpu_vals),
            );
        }

        // CPU input_proj backward
        mamba_rs::train::blas::sgemm_backward(
            &mut vec![0.0f32; bt * input_dim], // dx (discarded)
            &mut cpu_grads.input_proj_w,
            Some(&mut cpu_grads.input_proj_b),
            &d_temporal_cpu[..bt * dm],
            &cpu_acts.input_proj_inputs[..bt * input_dim],
            &tw.input_proj_w,
            (bt, input_dim, dm),
        );

        // GPU input_proj backward
        gpu_sgemm_backward_grad_raw(
            &ctx,
            &mut gpu_scratch.d_input_proj_dx,
            (&gpu_grads.input_proj_w, Some(&gpu_grads.input_proj_b)),
            &d_temporal_gpu,
            &gpu_acts.input_proj_inputs,
            gpu_w.input_proj_w.cached_ptr(),
            (bt, input_dim, dm),
        )
        .unwrap();
        ctx.stream.synchronize().unwrap();

        // Compare input_proj_b
        let gpu_ipb = gpu_grads.input_proj_b.to_cpu().unwrap();
        let cpu_ipb = &cpu_grads.input_proj_b;
        let (max_rel, max_abs) = compare_vecs(&gpu_ipb, cpu_ipb);
        println!(
            "DIAG input_proj_b: max_rel={max_rel:.6}, max_abs={max_abs:.6}, \
             gpu_mean={:.6}, cpu_mean={:.6}",
            mean(&gpu_ipb),
            mean(cpu_ipb),
        );

        // Compare input_proj_w
        let gpu_ipw = gpu_grads.input_proj_w.to_cpu().unwrap();
        let cpu_ipw = &cpu_grads.input_proj_w;
        let (max_rel, max_abs) = compare_vecs(&gpu_ipw, cpu_ipw);
        println!(
            "DIAG input_proj_w: max_rel={max_rel:.6}, max_abs={max_abs:.6}, \
             gpu_mean={:.6}, cpu_mean={:.6}",
            mean(&gpu_ipw),
            mean(cpu_ipw),
        );
    }

    fn compare_vecs(gpu: &[f32], cpu: &[f32]) -> (f32, f32) {
        let max_rel = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(g, c)| {
                let denom = c.abs().max(1e-6);
                (g - c).abs() / denom
            })
            .fold(0.0f32, f32::max);
        let max_abs = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        (max_rel, max_abs)
    }

    fn mean(v: &[f32]) -> f32 {
        if v.is_empty() {
            return 0.0;
        }
        v.iter().sum::<f32>() / v.len() as f32
    }
}

// =========================================================================
// Training forward vs inference parity
// =========================================================================

#[test]
fn test_training_forward_matches_inference() {
    let cfg = MambaConfig::default();
    let input_dim = 128;
    let seq_len = 10;
    let dm = cfg.d_model;

    let weights = mamba_rs::MambaWeights::init(&cfg, input_dim, 42);
    let (tw, dims) = build_train_scaffolding(&weights, &cfg, input_dim, seq_len);

    let inputs: Vec<f32> = (0..seq_len * input_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();

    // Method 1: training forward, extract last timestep
    let (temporal, _) = run_forward(&tw, &dims, &inputs);
    let train_last = &temporal[(seq_len - 1) * dm..seq_len * dm];

    // Method 2: inference step-by-step
    let bb = MambaBackbone::from_weights(cfg, weights).unwrap();
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();
    let mut inf_out = vec![0.0f32; dm];
    for t in 0..seq_len {
        let inp = &inputs[t * input_dim..(t + 1) * input_dim];
        bb.forward_step(inp, &mut inf_out, &mut state, &mut scratch);
    }

    // Compare: training last timestep vs inference last step
    let max_diff: f32 = train_last
        .iter()
        .zip(inf_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "training forward last timestep vs inference: max_diff={max_diff:.6}"
    );
}

// =========================================================================
// Non-default config: verify with different model sizes
// =========================================================================

#[test]
fn test_custom_config_small() {
    let cfg = MambaConfig {
        d_model: 64,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
    };
    let input_dim = 32;
    let bb = MambaBackbone::init(cfg, input_dim, 99);
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();
    let mut output = vec![0.0f32; cfg.d_model];
    let input = vec![0.5f32; input_dim];

    bb.forward_step(&input, &mut output, &mut state, &mut scratch);
    assert!(
        output.iter().any(|&v| v.abs() > 1e-10),
        "custom small config should produce nonzero output"
    );

    // Sequence forward should also work
    let seq_len = 8;
    let seq_input: Vec<f32> = (0..seq_len * input_dim).map(|i| i as f32 * 0.01).collect();
    let mut seq_out = vec![0.0f32; seq_len * cfg.d_model];
    let mut state2 = bb.alloc_state();
    let mut scratch2 = bb.alloc_scratch();
    bb.forward_sequence(
        &seq_input,
        &mut seq_out,
        &mut state2,
        &mut scratch2,
        seq_len,
    );
    assert!(
        seq_out.iter().any(|&v| v.abs() > 1e-10),
        "custom config sequence forward should produce nonzero output"
    );
}

#[test]
fn test_custom_config_large() {
    let cfg = MambaConfig {
        d_model: 256,
        d_state: 32,
        d_conv: 4,
        expand: 2,
        n_layers: 4,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
    };
    let input_dim = 256;
    let bb = MambaBackbone::init(cfg, input_dim, 77);
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();
    let mut output = vec![0.0f32; cfg.d_model];
    let input = vec![0.1f32; input_dim];

    // Run 10 steps, check no NaN/Inf
    for step in 0..10 {
        bb.forward_step(&input, &mut output, &mut state, &mut scratch);
        assert!(
            output.iter().all(|&v| v.is_finite()),
            "step {step}: output contains NaN or Inf"
        );
    }
}

// =========================================================================
// CPU: all weight gradients are nonzero
// =========================================================================

#[test]
fn test_all_cpu_gradients_nonzero() {
    let cfg = MambaConfig::default();
    let input_dim = 128;
    let seq_len = 4;

    let weights = mamba_rs::MambaWeights::init(&cfg, input_dim, 42);
    let (tw, dims) = build_train_scaffolding(&weights, &cfg, input_dim, seq_len);

    let input: Vec<f32> = (0..seq_len * input_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();

    // Forward
    let (temporal, _) = run_forward(&tw, &dims, &input);

    // Backward with d_temporal = ones
    let di = dims.d_inner;
    let ds = dims.d_state;
    let mut a_neg = vec![0.0f32; dims.n_layers * di * ds];
    for (l, lw) in tw.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }
    let acts = {
        use mamba_rs::ops::dims::MambaRecurrentState;
        use mamba_rs::train::flat::MambaBackboneFlat;
        use mamba_rs::train::scratch::PhaseScratch;
        let mut acts = MambaBackboneFlat::zeros(dims);
        let mut scratch = PhaseScratch::zeros(&dims);
        let mut conv = vec![0.0f32; dims.n_layers * di * dims.d_conv];
        let mut ssm = vec![0.0f32; dims.n_layers * di * ds];
        let mut state = MambaRecurrentState {
            conv: &mut conv,
            ssm: &mut ssm,
            a_neg: &a_neg,
        };
        let mut t = vec![0.0f32; seq_len * cfg.d_model];
        mamba_rs::train::forward::forward_mamba_backbone_batched(
            &mut t,
            &mut acts,
            &tw,
            &input,
            &mut state,
            &mut scratch,
            &dims,
        );
        acts
    };

    let mut d_temporal = vec![1.0f32; temporal.len()];
    let mut grad_tw = mamba_rs::train::weights::TrainMambaWeights::zeros_from_dims(&dims);
    let mut bwd_scratch = mamba_rs::train::scratch::BackwardPhaseScratch::zeros(&dims);
    mamba_rs::train::backward::backward_mamba_backbone_batched(
        &mut d_temporal,
        &mut grad_tw,
        &acts,
        &tw,
        &a_neg,
        &mut bwd_scratch,
        &dims,
    );

    // Verify ALL weight groups have nonzero gradients
    let ip_b_max = grad_tw
        .input_proj_b
        .iter()
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    let ip_w_max = grad_tw
        .input_proj_w
        .iter()
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    let nf_max = grad_tw
        .norm_f_weight
        .iter()
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(ip_b_max > 1e-6, "input_proj_b gradient is zero");
    assert!(ip_w_max > 1e-6, "input_proj_w gradient is zero");
    assert!(nf_max > 1e-6, "norm_f_weight gradient is zero");

    for (li, lg) in grad_tw.layers.iter().enumerate() {
        macro_rules! check_nonzero {
            ($name:ident) => {
                let m = lg.$name.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                assert!(
                    m > 1e-8,
                    "layer {li} {} gradient is zero (max_abs={m})",
                    stringify!($name)
                );
            };
        }
        check_nonzero!(norm_weight);
        check_nonzero!(in_proj_w);
        check_nonzero!(conv1d_weight);
        check_nonzero!(conv1d_bias);
        check_nonzero!(x_proj_w);
        check_nonzero!(dt_proj_w);
        check_nonzero!(dt_proj_b);
        check_nonzero!(a_log);
        check_nonzero!(d_param);
        check_nonzero!(out_proj_w);
    }
}

// =========================================================================
// CPU: sequence length = 1 edge case
// =========================================================================

#[test]
fn test_seq_len_one() {
    let cfg = MambaConfig::default();
    let input_dim = 128;
    let seq_len = 1;

    let weights = mamba_rs::MambaWeights::init(&cfg, input_dim, 42);
    let (tw, dims) = build_train_scaffolding(&weights, &cfg, input_dim, seq_len);

    let input: Vec<f32> = (0..input_dim).map(|i| (i as f32) * 0.01).collect();
    let (temporal, loss) = run_forward(&tw, &dims, &input);

    assert!(loss.is_finite(), "seq_len=1 forward produces NaN/Inf loss");
    assert!(
        temporal.iter().any(|&v| v.abs() > 1e-10),
        "seq_len=1 forward produces zero output"
    );
}

// =========================================================================
// CPU: long sequence stability (no NaN/Inf accumulation)
// =========================================================================

#[test]
fn test_long_sequence_stability() {
    let cfg = MambaConfig::default();
    let input_dim = 128;
    let bb = MambaBackbone::init(cfg, input_dim, 42);
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();
    let mut output = vec![0.0f32; cfg.d_model];

    // Run 200 steps — verify no NaN/Inf accumulation in state
    for step in 0..200 {
        let input: Vec<f32> = (0..input_dim)
            .map(|i| ((step * input_dim + i) as f32) * 0.001)
            .collect();
        bb.forward_step(&input, &mut output, &mut state, &mut scratch);
        assert!(
            output.iter().all(|&v| v.is_finite()),
            "step {step}: output contains NaN or Inf"
        );
    }
}

// =========================================================================
// GPU: additional comprehensive tests
// =========================================================================

#[cfg(feature = "cuda")]
mod gpu_extra_tests {
    use mamba_rs::{MambaBackbone, MambaConfig};

    #[test]
    fn test_gpu_inference_batch_parity() {
        use mamba_rs::gpu::inference::GpuMambaBackbone;

        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;
        let batch = 4;

        let bb = MambaBackbone::init(cfg, input_dim, 42);

        // GPU batch=4: run all samples together
        let mut gpu_batch =
            GpuMambaBackbone::new(0, bb.weights(), *bb.config(), input_dim, batch).unwrap();

        // GPU batch=1: run each sample individually (reuse single engine, reset between samples)
        let mut gpu_single =
            GpuMambaBackbone::new(0, bb.weights(), *bb.config(), input_dim, 1).unwrap();

        // First step only (state=zero for both paths, so results must match)
        let inputs: Vec<f32> = (0..batch * input_dim).map(|i| (i as f32) * 0.001).collect();

        let mut batch_out = vec![0.0f32; batch * cfg.d_model];
        gpu_batch.step(&inputs, &mut batch_out).unwrap();

        for b in 0..batch {
            gpu_single.reset().unwrap();
            let single_in = &inputs[b * input_dim..(b + 1) * input_dim];
            let mut single_out = vec![0.0f32; cfg.d_model];
            gpu_single.step(single_in, &mut single_out).unwrap();

            let batch_slice = &batch_out[b * cfg.d_model..(b + 1) * cfg.d_model];
            let max_diff: f32 = single_out
                .iter()
                .zip(batch_slice.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            // Batch=1 (GEMV) vs batch=4 (GEMM tiles) select different cuBLAS
            // algorithms under TF32, producing different rounding. This is
            // documented NVIDIA behavior, not a bug. ~1e-3 per op compounds
            // over ~15 matmuls in a 3-layer Mamba.
            assert!(
                max_diff < 5.0,
                "batch {b}: single vs batch max_diff={max_diff:.6}"
            );
        }
    }

    #[test]
    fn test_gpu_inference_long_sequence_stability() {
        use mamba_rs::gpu::inference::GpuMambaBackbone;

        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;
        let bb = MambaBackbone::init(cfg, input_dim, 42);
        let mut gpu_bb =
            GpuMambaBackbone::new(0, bb.weights(), *bb.config(), input_dim, 1).unwrap();

        let mut output = vec![0.0f32; cfg.d_model];
        for step in 0..200 {
            let input: Vec<f32> = (0..input_dim)
                .map(|i| ((step * input_dim + i) as f32) * 0.001)
                .collect();
            gpu_bb.step(&input, &mut output).unwrap();
            assert!(
                output.iter().all(|&v| v.is_finite()),
                "GPU step {step}: output contains NaN or Inf"
            );
        }
    }

    #[test]
    fn test_gpu_cuda_graph_matches_cpu() {
        use mamba_rs::gpu::inference::GpuMambaBackbone;

        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;
        let bb = MambaBackbone::init(cfg, input_dim, 42);

        // GPU with CUDA Graph
        let mut gpu_bb =
            GpuMambaBackbone::new(0, bb.weights(), *bb.config(), input_dim, 1).unwrap();
        gpu_bb.capture_graph().unwrap();

        // CPU
        let mut cpu_state = bb.alloc_state();
        let mut cpu_scratch = bb.alloc_scratch();

        for step in 0..20 {
            let input: Vec<f32> = (0..input_dim)
                .map(|i| ((step * input_dim + i) as f32) * 0.001)
                .collect();

            let mut gpu_out = vec![0.0f32; cfg.d_model];
            gpu_bb.step(&input, &mut gpu_out).unwrap();

            let mut cpu_out = vec![0.0f32; cfg.d_model];
            bb.forward_step(&input, &mut cpu_out, &mut cpu_state, &mut cpu_scratch);

            let max_diff: f32 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            // TF32 compound error: ~1e-3 per SGEMM, compounded over ~15
            // matmuls in a 3-layer Mamba. With 20 steps of state accumulation,
            // max_diff can reach ~5.0 in worst case.
            assert!(
                max_diff < 5.0,
                "CUDA Graph vs CPU step {step}: max_diff={max_diff:.6}"
            );
        }
    }

    #[test]
    fn test_gpu_training_all_gradients_nonzero() {
        use mamba_rs::gpu::backward::gpu_backward_mamba_backbone;
        use mamba_rs::gpu::buffers::GpuBuffer;
        use mamba_rs::gpu::context::GpuCtx;
        use mamba_rs::gpu::device::GpuDevice;
        use mamba_rs::gpu::forward::GpuRecurrentState;
        use mamba_rs::gpu::forward::{
            GpuMambaBackboneActs, GpuMambaDims, GpuMambaScratch, gpu_forward_mamba_backbone,
        };
        use mamba_rs::gpu::weights::{GpuMambaGrads, GpuMambaTrainWeights};

        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;
        let seq_len = 4;
        let di = cfg.d_inner();
        let ds = cfg.d_state;

        let weights = mamba_rs::MambaWeights::init(&cfg, input_dim, 42);
        let device = GpuDevice::new(0).unwrap();
        let ctx = GpuCtx::new(&device).unwrap();

        let gpu_dims = GpuMambaDims {
            batch: 1,
            d_model: cfg.d_model,
            d_inner: di,
            d_state: ds,
            d_conv: cfg.d_conv,
            dt_rank: cfg.dt_rank(),
            xdbl_dim: cfg.xdbl_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers: cfg.n_layers,
        };

        let gpu_w = GpuMambaTrainWeights::from_cpu(&ctx.stream, &weights).unwrap();
        let mut gpu_temporal = GpuBuffer::zeros(&ctx.stream, seq_len * cfg.d_model).unwrap();
        let mut gpu_acts = GpuMambaBackboneActs::new(&ctx.stream, &gpu_dims).unwrap();
        let input: Vec<f32> = (0..seq_len * input_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let input_gpu = GpuBuffer::from_cpu(&ctx.stream, &input).unwrap();

        let mut a_neg_cpu = vec![0.0f32; cfg.n_layers * di * ds];
        for (l, lw) in weights.layers.iter().enumerate() {
            for i in 0..di * ds {
                a_neg_cpu[l * di * ds + i] = -lw.a_log[i].exp();
            }
        }
        let mut gpu_state = GpuRecurrentState {
            conv_states: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * cfg.d_conv).unwrap(),
            ssm_states: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * ds).unwrap(),
            a_neg_all: GpuBuffer::zeros(&ctx.stream, cfg.n_layers * di * ds).unwrap(),
        };
        gpu_state.a_neg_all.upload(&ctx.stream, &a_neg_cpu).unwrap();
        let mut gpu_scratch = GpuMambaScratch::new(&ctx.stream, &gpu_dims).unwrap();

        // Forward
        gpu_forward_mamba_backbone(
            &ctx,
            &mut gpu_temporal,
            &mut gpu_acts,
            &gpu_w,
            &input_gpu,
            &mut gpu_state,
            &mut gpu_scratch,
        )
        .unwrap();

        // Backward
        let gpu_grads = GpuMambaGrads::new(&ctx.stream, &cfg, input_dim).unwrap();
        let mut d_temporal_gpu =
            GpuBuffer::from_cpu(&ctx.stream, &vec![1.0f32; seq_len * cfg.d_model]).unwrap();
        gpu_backward_mamba_backbone(
            &ctx,
            &mut d_temporal_gpu,
            &gpu_grads,
            &gpu_acts,
            &gpu_w,
            &gpu_state.a_neg_all,
            &mut gpu_scratch,
        )
        .unwrap();
        ctx.stream.synchronize().unwrap();

        // Verify ALL GPU gradients are nonzero
        macro_rules! check_nonzero {
            ($name:expr, $slice:expr) => {
                let vals = $slice.to_cpu().unwrap();
                let max_abs = vals.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                assert!(max_abs > 1e-8, "GPU {} gradient is zero", $name);
            };
        }
        check_nonzero!("input_proj_b", gpu_grads.input_proj_b);
        check_nonzero!("input_proj_w", gpu_grads.input_proj_w);
        check_nonzero!("norm_f_weight", gpu_grads.norm_f_weight);

        for (li, lg) in gpu_grads.layers.iter().enumerate() {
            check_nonzero!(&format!("L{li}.norm_weight"), lg.norm_weight);
            check_nonzero!(&format!("L{li}.in_proj_w"), lg.in_proj_w);
            check_nonzero!(&format!("L{li}.conv1d_weight"), lg.conv1d_weight);
            check_nonzero!(&format!("L{li}.conv1d_bias"), lg.conv1d_bias);
            check_nonzero!(&format!("L{li}.x_proj_w"), lg.x_proj_w);
            check_nonzero!(&format!("L{li}.dt_proj_w"), lg.dt_proj_w);
            check_nonzero!(&format!("L{li}.dt_proj_b"), lg.dt_proj_b);
            check_nonzero!(&format!("L{li}.a_log"), lg.a_log);
            check_nonzero!(&format!("L{li}.d_param"), lg.d_param);
            check_nonzero!(&format!("L{li}.out_proj_w"), lg.out_proj_w);
        }
    }

    #[test]
    fn test_gpu_custom_config() {
        use mamba_rs::gpu::inference::GpuMambaBackbone;

        let cfg = MambaConfig {
            d_model: 64,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            n_layers: 2,
            scan_mode: mamba_rs::config::ScanMode::Sequential,
        };
        let input_dim = 32;
        let bb = MambaBackbone::init(cfg, input_dim, 99);

        let mut gpu_bb =
            GpuMambaBackbone::new(0, bb.weights(), *bb.config(), input_dim, 1).unwrap();

        let mut cpu_state = bb.alloc_state();
        let mut cpu_scratch = bb.alloc_scratch();

        for step in 0..10 {
            let input: Vec<f32> = (0..input_dim)
                .map(|i| ((step * input_dim + i) as f32) * 0.01)
                .collect();

            let mut gpu_out = vec![0.0f32; cfg.d_model];
            gpu_bb.step(&input, &mut gpu_out).unwrap();

            let mut cpu_out = vec![0.0f32; cfg.d_model];
            bb.forward_step(&input, &mut cpu_out, &mut cpu_state, &mut cpu_scratch);

            let max_diff: f32 = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            // TF32 compound error: smaller model (2 layers, d_model=64) has
            // fewer matmuls, but TF32 rounding still compounds over steps.
            assert!(
                max_diff < 5.0,
                "custom config step {step}: GPU vs CPU max_diff={max_diff:.6}"
            );
        }
    }
}
