//! Full-sequence CPU prefill contracts.
//!
//! - bitwise anchor vs the training forward (same GEMMs, same fast_exp,
//!   same per-element chains);
//! - Parallel == Single bitwise (channel-independent phases, tile-local
//!   GEMM accumulation);
//! - prefill-then-decode handoff into the per-step inference path;
//! - identity input_proj branch == explicit eye projection, bitwise;
//! - batch helper == sequential prefills, bitwise.

use mamba_rs::config::MambaConfig;
use mamba_rs::inference::{
    PrefillMode, PrefillScratch, forward_mamba_backbone_prefill,
    forward_mamba_backbone_prefill_mode, prefill_batch,
};
use mamba_rs::module::MambaBackbone;
use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::state::MambaState;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::PhaseScratch;
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::weights::MambaWeights;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
        })
        .collect()
}

fn test_cfg() -> MambaConfig {
    MambaConfig {
        d_model: 32,
        n_layers: 2,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    }
}

fn init_weights(cfg: &MambaConfig, input_dim: usize, seed: u64) -> MambaWeights {
    let mut w = MambaWeights::init(cfg, input_dim, seed);
    for lw in w.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    w
}

fn train_weights_from(w: &MambaWeights) -> TrainMambaWeights {
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

fn assert_bits(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label}[{i}]: {x} vs {y}");
    }
}

/// Prefill (zero state) must reproduce the training forward bit-for-bit.
#[test]
fn prefill_matches_training_forward_bitwise() {
    let cfg = test_cfg();
    let input_dim = 20usize;
    let seq_len = 8usize;
    let w = init_weights(&cfg, input_dim, 0xC0FFEE);
    let input = det(seq_len * input_dim, 0xAA, 0.05);
    let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
    let di = cfg.d_inner();
    let (ds, dc, nl) = (cfg.d_state, cfg.d_conv, cfg.n_layers);

    // Training forward.
    let tw = train_weights_from(&w);
    let mut a_neg = vec![0.0f32; nl * di * ds];
    for (l, lw) in w.layers.iter().enumerate() {
        a_neg[l * di * ds..(l + 1) * di * ds].copy_from_slice(&lw.a_neg);
    }
    let mut acts = MambaBackboneFlat::zeros(dims);
    let mut fwd_scratch = PhaseScratch::zeros(&dims);
    let mut conv = vec![0.0f32; nl * di * dc];
    let mut ssm = vec![0.0f32; nl * di * ds];
    let mut tr_state = MambaRecurrentState {
        conv: &mut conv,
        ssm: &mut ssm,
        a_neg: &a_neg,
    };
    let mut train_out = vec![0.0f32; seq_len * cfg.d_model];
    forward_mamba_backbone_batched(
        &mut train_out,
        &mut acts,
        &tw,
        &input,
        &mut tr_state,
        &mut fwd_scratch,
        &dims,
    );

    // Prefill.
    let mut state = MambaState::zeros(nl, di, ds, dc);
    let mut scratch = PrefillScratch::new(&dims);
    let mut out = vec![0.0f32; seq_len * cfg.d_model];
    forward_mamba_backbone_prefill(&mut out, &input, &w, &mut state, &mut scratch, &dims);

    assert_bits("prefill vs training forward", &out, &train_out);
}

/// Parallel mode must be bit-equal to Single.
#[test]
fn prefill_parallel_bit_equals_single() {
    let cfg = test_cfg();
    let input_dim = 20usize;
    let seq_len = 16usize;
    let w = init_weights(&cfg, input_dim, 0xC0FFEE);
    let input = det(seq_len * input_dim, 0xAB, 0.05);
    let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
    let (di, ds, dc, nl) = (cfg.d_inner(), cfg.d_state, cfg.d_conv, cfg.n_layers);

    let run = |mode: PrefillMode| -> (Vec<f32>, MambaState) {
        let mut state = MambaState::zeros(nl, di, ds, dc);
        let mut scratch = PrefillScratch::new(&dims);
        let mut out = vec![0.0f32; seq_len * cfg.d_model];
        forward_mamba_backbone_prefill_mode(
            &mut out,
            &input,
            &w,
            &mut state,
            &mut scratch,
            &dims,
            mode,
        );
        (out, state)
    };
    let (out_s, st_s) = run(PrefillMode::Single);
    let (out_p, st_p) = run(PrefillMode::Parallel);
    assert_bits("parallel vs single output", &out_p, &out_s);
    for (l, (a, b)) in st_s.layers.iter().zip(st_p.layers.iter()).enumerate() {
        assert_bits(&format!("L{l} conv_state"), &b.conv_state, &a.conv_state);
        assert_bits(&format!("L{l} ssm_state"), &b.ssm_state, &a.ssm_state);
    }
}

/// After prefilling T tokens, the per-step decode path must continue the
/// sequence as if all T+1 tokens had gone through the step loop (tolerance:
/// the step path uses matvec accumulation orders, prefill uses SGEMM).
#[test]
fn prefill_then_decode_handoff() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model; // backbone from_weights derives input_dim
    let seq_len = 8usize;
    let w = init_weights(&cfg, input_dim, 0xC0FFEE);
    let backbone = MambaBackbone::from_weights(cfg, w.clone()).expect("backbone");
    let inputs = det((seq_len + 1) * input_dim, 0xAC, 0.05);

    // Reference: the pure step loop over T+1 tokens.
    let mut ref_state = backbone.alloc_state();
    let mut ref_scratch = backbone.alloc_scratch();
    let mut ref_out = vec![0.0f32; (seq_len + 1) * cfg.d_model];
    backbone.forward_sequence(
        &inputs,
        &mut ref_out,
        &mut ref_state,
        &mut ref_scratch,
        seq_len + 1,
    );

    // Prefill T, then decode token T via the step path.
    let mut state = backbone.alloc_state();
    let mut prefill_scratch = backbone.alloc_prefill_scratch(seq_len);
    let mut prefill_out = vec![0.0f32; seq_len * cfg.d_model];
    backbone.forward_prefill(
        &inputs[..seq_len * input_dim],
        &mut prefill_out,
        &mut state,
        &mut prefill_scratch,
        seq_len,
        PrefillMode::Single,
    );
    let mut step_scratch = backbone.alloc_scratch();
    let mut step_out = vec![0.0f32; cfg.d_model];
    backbone.forward_step(
        &inputs[seq_len * input_dim..],
        &mut step_out,
        &mut state,
        &mut step_scratch,
    );

    let ref_last = &ref_out[seq_len * cfg.d_model..];
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut worst = 0.0f32;
    for (&a, &b) in step_out.iter().zip(ref_last.iter()) {
        dot += a as f64 * b as f64;
        na += (a as f64).powi(2);
        nb += (b as f64).powi(2);
        let d = (a - b).abs();
        worst = worst.max(d / a.abs().max(b.abs()).max(1e-3));
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    assert!(
        cos > 0.9999 && worst < 1e-2,
        "handoff decode diverges from the step loop: cos={cos} max_rel={worst}"
    );
}

/// The identity branch must equal an explicit eye projection bit-for-bit
/// (X @ I + 0 is exact in f32).
#[test]
fn prefill_identity_equals_eye_projection() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let seq_len = 8usize;
    let mut w_eye = init_weights(&cfg, input_dim, 0xC0FFEE);
    w_eye.input_proj_w = (0..cfg.d_model * cfg.d_model)
        .map(|i| {
            if i / cfg.d_model == i % cfg.d_model {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    w_eye.input_proj_b = vec![0.0; cfg.d_model];
    let mut w_id = w_eye.clone();
    w_id.input_proj_w.clear();
    w_id.input_proj_b.clear();

    let input = det(seq_len * input_dim, 0xAD, 0.05);
    let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
    let (di, ds, dc, nl) = (cfg.d_inner(), cfg.d_state, cfg.d_conv, cfg.n_layers);

    let run = |w: &MambaWeights| -> Vec<f32> {
        let mut state = MambaState::zeros(nl, di, ds, dc);
        let mut scratch = PrefillScratch::new(&dims);
        let mut out = vec![0.0f32; seq_len * cfg.d_model];
        forward_mamba_backbone_prefill(&mut out, &input, w, &mut state, &mut scratch, &dims);
        out
    };
    assert_bits("identity vs eye", &run(&w_id), &run(&w_eye));
}

/// The batch helper must equal per-sample sequential prefills bit-for-bit.
#[test]
fn prefill_batch_matches_sequential() {
    let cfg = test_cfg();
    let input_dim = 20usize;
    let seq_len = 8usize;
    let b = 3usize;
    let w = init_weights(&cfg, input_dim, 0xC0FFEE);
    let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
    let (di, ds, dc, nl) = (cfg.d_inner(), cfg.d_state, cfg.d_conv, cfg.n_layers);
    let inputs = det(b * seq_len * input_dim, 0xAE, 0.05);

    // Sequential reference.
    let mut seq_out = vec![0.0f32; b * seq_len * cfg.d_model];
    for i in 0..b {
        let mut state = MambaState::zeros(nl, di, ds, dc);
        let mut scratch = PrefillScratch::new(&dims);
        forward_mamba_backbone_prefill(
            &mut seq_out[i * seq_len * cfg.d_model..(i + 1) * seq_len * cfg.d_model],
            &inputs[i * seq_len * input_dim..(i + 1) * seq_len * input_dim],
            &w,
            &mut state,
            &mut scratch,
            &dims,
        );
    }

    // Batch helper.
    let mut states: Vec<MambaState> = (0..b).map(|_| MambaState::zeros(nl, di, ds, dc)).collect();
    let mut scratches: Vec<PrefillScratch> = (0..b).map(|_| PrefillScratch::new(&dims)).collect();
    let mut batch_out = vec![0.0f32; b * seq_len * cfg.d_model];
    prefill_batch(
        &mut batch_out,
        &inputs,
        &w,
        &mut states,
        &mut scratches,
        &dims,
    );

    assert_bits("batch vs sequential", &batch_out, &seq_out);
}

/// Latency probe at the classifier center shape (manual: --ignored).
/// Prints ms/page for Single and Parallel modes plus the per-step loop
/// extrapolation — the first measured numbers for the doc-101 serve tier.
#[test]
#[ignore]
fn prefill_bench_classifier_shape() {
    let cfg = MambaConfig {
        d_model: 384,
        n_layers: 24,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    };
    let input_dim = 1024usize;
    // Production patchify T_TOTAL: 57x81 patches + 4 register tokens
    // (wellwon_classify src/patchify.rs).
    let seq_len = 4621usize;
    let w = init_weights(&cfg, input_dim, 0xC0FFEE);
    let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
    let (di, ds, dc, nl) = (cfg.d_inner(), cfg.d_state, cfg.d_conv, cfg.n_layers);
    let input = det(seq_len * input_dim, 0xAF, 0.05);
    let mut out = vec![0.0f32; seq_len * cfg.d_model];

    for mode in [PrefillMode::Single, PrefillMode::Parallel] {
        let mut state = MambaState::zeros(nl, di, ds, dc);
        let mut scratch = PrefillScratch::new(&dims);
        // Warm-up.
        forward_mamba_backbone_prefill_mode(
            &mut out,
            &input,
            &w,
            &mut state,
            &mut scratch,
            &dims,
            mode,
        );
        let reps = 3;
        let start = std::time::Instant::now();
        for _ in 0..reps {
            state.reset();
            forward_mamba_backbone_prefill_mode(
                &mut out,
                &input,
                &w,
                &mut state,
                &mut scratch,
                &dims,
                mode,
            );
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        eprintln!("prefill {mode:?}: {ms:.1} ms/page (T={seq_len}, d384x24)");
    }
}
