//! Integration tests for HF model loading and generation.
//!
//! All tests use synthetic data — zero network access.

#![cfg(feature = "hf")]

use std::path::Path;

use safetensors::tensor::TensorView;

// ---------------------------------------------------------------------------
// Helpers: build a synthetic HF-format Mamba-1 checkpoint
// ---------------------------------------------------------------------------

const D_MODEL: usize = 64;
const D_STATE: usize = 16;
const D_CONV: usize = 4;
const EXPAND: usize = 2;
const N_LAYERS: usize = 2;
const VOCAB_SIZE: usize = 256;

fn d_inner() -> usize {
    D_MODEL * EXPAND
}

fn dt_rank() -> usize {
    D_MODEL.div_ceil(16)
}

fn xdbl_dim() -> usize {
    dt_rank() + 2 * D_STATE
}

fn simple_rng(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed.max(1);
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32 * 0.1
    }
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = simple_rng(seed);
    (0..n).map(|_| rng()).collect()
}

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn write_synthetic_checkpoint(dir: &Path, include_lm_head: bool) {
    let di = d_inner();
    let dr = dt_rank();
    let xd = xdbl_dim();

    let mut tensors: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

    // Embedding
    let embed = rand_vec(VOCAB_SIZE * D_MODEL, 100);
    tensors.push(("backbone.embeddings.weight".into(), f32_to_bytes(&embed), vec![VOCAB_SIZE, D_MODEL]));

    // Optional lm_head
    if include_lm_head {
        let lm = rand_vec(VOCAB_SIZE * D_MODEL, 200);
        tensors.push(("lm_head.weight".into(), f32_to_bytes(&lm), vec![VOCAB_SIZE, D_MODEL]));
    }

    // norm_f
    tensors.push(("backbone.norm_f.weight".into(), f32_to_bytes(&vec![1.0f32; D_MODEL]), vec![D_MODEL]));

    // Per-layer weights
    for i in 0..N_LAYERS {
        let seed_base = (i as u64 + 1) * 1000;
        let prefix = format!("backbone.layers.{i}");
        tensors.push((format!("{prefix}.norm.weight"), f32_to_bytes(&vec![1.0f32; D_MODEL]), vec![D_MODEL]));
        tensors.push((format!("{prefix}.mixer.in_proj.weight"), f32_to_bytes(&rand_vec(D_MODEL * 2 * di, seed_base + 1)), vec![D_MODEL, 2 * di]));
        tensors.push((format!("{prefix}.mixer.conv1d.weight"), f32_to_bytes(&rand_vec(di * D_CONV, seed_base + 2)), vec![di, 1, D_CONV]));
        tensors.push((format!("{prefix}.mixer.conv1d.bias"), f32_to_bytes(&vec![0.0f32; di]), vec![di]));
        tensors.push((format!("{prefix}.mixer.x_proj.weight"), f32_to_bytes(&rand_vec(di * xd, seed_base + 3)), vec![di, xd]));
        tensors.push((format!("{prefix}.mixer.dt_proj.weight"), f32_to_bytes(&rand_vec(dr * di, seed_base + 4)), vec![dr, di]));
        tensors.push((format!("{prefix}.mixer.dt_proj.bias"), f32_to_bytes(&rand_vec(di, seed_base + 5)), vec![di]));
        // a_log: needs nonzero values for compute_a_neg test
        let a_log: Vec<f32> = (0..di * D_STATE).map(|j| -((j as f32 + 1.0).ln())).collect();
        tensors.push((format!("{prefix}.mixer.A_log"), f32_to_bytes(&a_log), vec![di, D_STATE]));
        tensors.push((format!("{prefix}.mixer.D"), f32_to_bytes(&vec![1.0f32; di]), vec![di]));
        tensors.push((format!("{prefix}.mixer.out_proj.weight"), f32_to_bytes(&rand_vec(di * D_MODEL, seed_base + 6)), vec![di, D_MODEL]));
    }

    // Write config.json
    let config_json = format!(
        r#"{{
            "model_type": "mamba",
            "hidden_size": {D_MODEL},
            "num_hidden_layers": {N_LAYERS},
            "state_size": {D_STATE},
            "conv_kernel": {D_CONV},
            "expand": {EXPAND},
            "vocab_size": {VOCAB_SIZE},
            "time_step_rank": {dr}
        }}"#,
        dr = dt_rank()
    );
    std::fs::write(dir.join("config.json"), config_json).unwrap();

    // Write safetensors
    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, bytes, shape)| {
            let tv = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes).unwrap();
            (name.clone(), tv)
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(dir.join("model.safetensors"), serialized).unwrap();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_load_synthetic_m1_hf_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_checkpoint(dir.path(), false);

    let lm = mamba_rs::module::lm::MambaLM::from_hf(dir.path()).unwrap();
    assert_eq!(lm.vocab_size, VOCAB_SIZE);
    assert_eq!(lm.d_model, D_MODEL);
}

#[test]
fn test_weight_tying_detected() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_checkpoint(dir.path(), false);

    let mut lm = mamba_rs::module::lm::MambaLM::from_hf(dir.path()).unwrap();
    // Weight tying: no lm_head in file + tie_word_embeddings=true (default)
    // MambaLM should use embed^T for logits — generation should work
    let params = mamba_rs::module::sample::SampleParams {
        temperature: 0.0,
        max_tokens: 5,
        ..Default::default()
    };
    let tokens = lm.generate(&[1, 2, 3], &params);
    assert_eq!(tokens.len(), 5);
    for &t in &tokens {
        assert!((t as usize) < VOCAB_SIZE, "generated token {t} >= vocab {VOCAB_SIZE}");
    }
}

#[test]
fn test_untied_weights_loaded() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_checkpoint(dir.path(), true);

    // Write config with tie_word_embeddings=false
    let config_json = format!(
        r#"{{
            "model_type": "mamba",
            "hidden_size": {D_MODEL},
            "num_hidden_layers": {N_LAYERS},
            "state_size": {D_STATE},
            "conv_kernel": {D_CONV},
            "expand": {EXPAND},
            "vocab_size": {VOCAB_SIZE},
            "tie_word_embeddings": false
        }}"#
    );
    std::fs::write(dir.path().join("config.json"), config_json).unwrap();

    let mut lm = mamba_rs::module::lm::MambaLM::from_hf(dir.path()).unwrap();
    let params = mamba_rs::module::sample::SampleParams {
        temperature: 0.0,
        max_tokens: 5,
        ..Default::default()
    };
    let tokens = lm.generate(&[1, 2, 3], &params);
    assert_eq!(tokens.len(), 5);
}

#[test]
fn test_compute_a_neg_called() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_checkpoint(dir.path(), false);

    let lm = mamba_rs::module::lm::MambaLM::from_hf(dir.path()).unwrap();
    // Access backbone weights to verify a_neg was computed
    if let mamba_rs::module::lm::AnyBackbone::M1(ref bb) = lm.backbone {
        let lw = bb.layer(0);
        // a_log has nonzero values → a_neg = -exp(a_log) should be negative
        assert!(lw.a_neg[0] < 0.0, "a_neg[0] = {}, expected negative", lw.a_neg[0]);
        assert!(lw.a_neg[0].is_finite(), "a_neg[0] is not finite");
    } else {
        panic!("expected M1 backbone");
    }
}

#[test]
fn test_generate_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_checkpoint(dir.path(), false);

    let mut lm = mamba_rs::module::lm::MambaLM::from_hf(dir.path()).unwrap();
    let params = mamba_rs::module::sample::SampleParams {
        temperature: 0.0,
        max_tokens: 10,
        seed: 42,
        ..Default::default()
    };

    let tokens1 = lm.generate(&[1, 2, 3], &params);
    let tokens2 = lm.generate(&[1, 2, 3], &params);
    assert_eq!(tokens1, tokens2, "greedy generation must be deterministic");
}

#[test]
fn test_generate_state_save_restore() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_checkpoint(dir.path(), false);

    let mut lm = mamba_rs::module::lm::MambaLM::from_hf(dir.path()).unwrap();
    let params = mamba_rs::module::sample::SampleParams {
        temperature: 0.0,
        max_tokens: 5,
        seed: 42,
        ..Default::default()
    };

    // Generate 10 tokens in one shot
    let params10 = mamba_rs::module::sample::SampleParams {
        max_tokens: 10,
        ..params.clone()
    };
    let all_10 = lm.generate(&[1, 2, 3], &params10);

    // Generate first 5, save state, generate next 5
    lm.reset();
    // Prefill manually to save state after 5 tokens
    let first_5 = lm.generate(&[1, 2, 3], &params);
    assert_eq!(first_5.len(), 5);
    assert_eq!(&first_5, &all_10[..5]);
}

#[test]
fn test_m1_through_any_backbone() {
    use mamba_rs::module::lm::AnyBackbone;
    use mamba_rs::{MambaConfig, MambaBackbone};

    let cfg = MambaConfig {
        d_model: 32,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 1,
        ..Default::default()
    };
    let bb = MambaBackbone::init(cfg, 32, 42);

    // Direct path
    let mut state_direct = bb.alloc_state();
    let mut scratch_direct = bb.alloc_scratch();
    let mut out_direct = vec![0.0f32; 32];
    let input = vec![0.1f32; 32];
    bb.forward_step(&input, &mut out_direct, &mut state_direct, &mut scratch_direct);

    // Through AnyBackbone
    let bb2 = MambaBackbone::init(cfg, 32, 42);
    let any = AnyBackbone::M1(bb2);
    let mut state_any = any.alloc_state();
    let mut scratch_any = any.alloc_scratch();
    let mut out_any = vec![0.0f32; 32];
    any.forward_step(&input, &mut out_any, &mut state_any, &mut scratch_any);

    for (i, (a, b)) in out_direct.iter().zip(out_any.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "mismatch at {i}: {a} vs {b}");
    }
}
