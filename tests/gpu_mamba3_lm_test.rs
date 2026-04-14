//! GpuMamba3LM smoke tests — synthetic weights (no HF M3 checkpoint exists).

#![cfg(all(feature = "cuda", feature = "hf"))]

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::module::gpu_lm3::GpuMamba3LM;
use mamba_rs::module::sample::SampleParams;

fn small_m3_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: 64,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
    }
}

fn build_synthetic_lm(dtype: WeightDtype, vocab_size: usize) -> GpuMamba3LM {
    let cfg = small_m3_cfg();
    let input_dim = cfg.d_model;
    let weights = Mamba3Weights::init(&cfg, input_dim, 0xDEADBEEF);
    // Tied lm_head: embed table is [vocab_size, d_model].
    let d = cfg.d_model;
    let mut embed = vec![0.0f32; vocab_size * d];
    let mut seed: u64 = 0xFACE;
    for v in embed.iter_mut() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        *v = (((seed & 0xFFFFFF) as f32 / 16777216.0) - 0.5) * 0.1;
    }
    GpuMamba3LM::from_weights_with_dtype(&weights, cfg, embed, None, vocab_size, 0, dtype).unwrap()
}

#[test]
fn test_gpu_mamba3_lm_f32_generates_deterministic() {
    let vocab_size = 32;
    let mut lm = build_synthetic_lm(WeightDtype::F32, vocab_size);
    assert_eq!(lm.dtype(), WeightDtype::F32);
    assert_eq!(lm.vocab_size, vocab_size);
    assert_eq!(lm.d_model, 64);

    let params = SampleParams {
        temperature: 0.0,
        max_tokens: 10,
        ..Default::default()
    };
    let tokens = lm.generate(&[1, 2, 3], &params).unwrap();
    assert_eq!(tokens.len(), 10, "expected 10 generated tokens");
    for &t in &tokens {
        assert!((t as usize) < vocab_size, "token {t} out of vocab");
    }

    // Determinism across resets.
    lm.reset().unwrap();
    let tokens2 = lm.generate(&[1, 2, 3], &params).unwrap();
    assert_eq!(tokens, tokens2, "f32 greedy should be deterministic");
}

#[test]
fn test_gpu_mamba3_lm_bf16_generates_and_matches_f32_structure() {
    let vocab_size = 32;
    let mut lm_f32 = build_synthetic_lm(WeightDtype::F32, vocab_size);
    let mut lm_bf16 = build_synthetic_lm(WeightDtype::Bf16, vocab_size);
    assert_eq!(lm_f32.dtype(), WeightDtype::F32);
    assert_eq!(lm_bf16.dtype(), WeightDtype::Bf16);

    let params = SampleParams {
        temperature: 0.0,
        max_tokens: 5,
        ..Default::default()
    };
    let toks_f32 = lm_f32.generate(&[1, 2, 3], &params).unwrap();
    let toks_bf16 = lm_bf16.generate(&[1, 2, 3], &params).unwrap();

    // Both backends must produce valid tokens.
    assert_eq!(toks_f32.len(), 5);
    assert_eq!(toks_bf16.len(), 5);
    for &t in &toks_f32 {
        assert!((t as usize) < vocab_size);
    }
    for &t in &toks_bf16 {
        assert!((t as usize) < vocab_size);
    }
    eprintln!("M3 LM f32:  {toks_f32:?}");
    eprintln!("M3 LM bf16: {toks_bf16:?}");

    // We don't assert token-for-token match on random-init synthetic weights
    // (first-step bf16 noise amplified by greedy argmax can pick different
    // tokens). The structural assertion — bf16 backend generates finite,
    // in-vocab tokens deterministically — is the smoke test signal.
    lm_bf16.reset().unwrap();
    let toks_bf16_again = lm_bf16.generate(&[1, 2, 3], &params).unwrap();
    assert_eq!(
        toks_bf16, toks_bf16_again,
        "bf16 greedy should be deterministic"
    );
}
