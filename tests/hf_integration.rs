//! Integration tests for HF model loading and generation.
//!
//! All tests use synthetic data — zero network access.

#![cfg(feature = "hf")]

#[path = "common/hf_synthetic.rs"]
mod hf_synthetic;

use hf_synthetic::{
    D_CONV, D_MODEL, D_STATE, EXPAND, N_LAYERS, VOCAB_SIZE, write_synthetic_checkpoint,
};

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
        assert!(
            (t as usize) < VOCAB_SIZE,
            "generated token {t} >= vocab {VOCAB_SIZE}"
        );
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
        assert!(
            lw.a_neg[0] < 0.0,
            "a_neg[0] = {}, expected negative",
            lw.a_neg[0]
        );
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
    use mamba_rs::{MambaBackbone, MambaConfig};

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
    bb.forward_step(
        &input,
        &mut out_direct,
        &mut state_direct,
        &mut scratch_direct,
    );

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
