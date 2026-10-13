//! The language-model checkpoint: a model of every family, with a tied or
//! an untied head, saved to one file loads back as the same model; a
//! backbone checkpoint is not a language model; and a Mamba-1 checkpoint
//! refuses to be read as another family's.
#![cfg(feature = "hf")]

use std::path::PathBuf;

use mamba_rs::config::ScanMode;
use mamba_rs::{
    Mamba2Backbone, Mamba2Config, Mamba2Weights, Mamba3Backbone, Mamba3Config, Mamba3MimoBackbone,
    Mamba3MimoConfig, Mamba3MimoWeights, Mamba3Weights, MambaBackbone, MambaConfig, MambaLM,
    MambaWeights, SampleParams,
};

const D_MODEL: usize = 32;
const VOCAB: usize = 90;
/// The embedding rows, the vocabulary padded the way the loaders pad.
const VOCAB_ROWS: usize = 96;

fn m1_cfg() -> MambaConfig {
    MambaConfig {
        d_model: D_MODEL,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
    }
}

fn m2_cfg() -> Mamba2Config {
    Mamba2Config {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba2Config::default()
    }
}

fn siso_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: D_MODEL,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        ..Mamba3Config::default()
    }
}

fn mimo_cfg() -> Mamba3MimoConfig {
    Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    }
}

/// Deterministic values in [-1, 1).
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "mamba_rs_lm_checkpoint_{}_{name}.safetensors",
        std::process::id()
    ))
}

/// A model of every family: Mamba-1 reads its embeddings as they are,
/// the others through an input projection; Mamba-2 and SISO have an
/// untied head.
fn models() -> Vec<MambaLM> {
    let embed = values(VOCAB_ROWS * D_MODEL, 1);
    let head = || Some(values(D_MODEL * VOCAB, 2));
    let mut m1 = MambaWeights::init(&m1_cfg(), D_MODEL, 3);
    m1.input_proj_w.clear();
    m1.input_proj_b.clear();
    vec![
        MambaLM::new(
            MambaBackbone::new(m1, m1_cfg(), D_MODEL).unwrap(),
            embed.clone(),
            None,
            VOCAB,
        ),
        MambaLM::new(
            Mamba2Backbone::new(
                Mamba2Weights::init(&m2_cfg(), D_MODEL, 4),
                m2_cfg(),
                D_MODEL,
            )
            .unwrap(),
            embed.clone(),
            head(),
            VOCAB,
        ),
        MambaLM::new(
            Mamba3Backbone::new(
                Mamba3Weights::init(&siso_cfg(), D_MODEL, 5),
                siso_cfg(),
                D_MODEL,
            )
            .unwrap(),
            embed.clone(),
            head(),
            VOCAB,
        ),
        MambaLM::new(
            Mamba3MimoBackbone::new(
                Mamba3MimoWeights::init(&mimo_cfg(), D_MODEL, 6),
                mimo_cfg(),
                D_MODEL,
            )
            .unwrap(),
            embed,
            None,
            VOCAB,
        ),
    ]
    .into_iter()
    .map(Result::unwrap)
    .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A model saved and loaded back generates the same tokens and gives the
/// same logits, bit for bit, in every family.
#[test]
fn every_family_round_trips_through_one_file() {
    let params = SampleParams {
        temperature: 0.0,
        max_tokens: 6,
        ..SampleParams::default()
    };
    let prompt: Vec<u32> = (0..20).map(|i| (i * 37 % VOCAB) as u32).collect();
    for mut lm in models() {
        let family = lm.family();
        let path = temp(&format!("{family:?}"));
        lm.save_checkpoint(&path).unwrap();
        let loaded = MambaLM::from_checkpoint(&path);
        std::fs::remove_file(&path).ok();
        let mut loaded = loaded.unwrap();
        assert_eq!(loaded.family(), family);
        assert_eq!(
            (loaded.vocab_size, loaded.d_model),
            (VOCAB, D_MODEL),
            "{family:?}"
        );
        let want = lm.generate(&prompt, &params);
        assert_eq!(want.len(), 6, "{family:?}");
        assert_eq!(loaded.generate(&prompt, &params), want, "{family:?}");
        assert_eq!(
            bits(loaded.last_logits(0)),
            bits(lm.last_logits(0)),
            "{family:?}"
        );
    }
}

/// A backbone checkpoint carries no embedding: loading it as a language
/// model is refused by name rather than failing on a missing tensor.
#[test]
fn a_backbone_checkpoint_is_not_a_language_model() {
    let path = temp("backbone");
    let w = Mamba2Weights::init(&m2_cfg(), D_MODEL, 7);
    mamba_rs::mamba2::serialize::save(&path, &w, &m2_cfg(), D_MODEL).unwrap();
    let got = MambaLM::from_checkpoint(&path);
    std::fs::remove_file(&path).ok();
    let err = got
        .err()
        .expect("a backbone checkpoint loaded as a language model");
    assert!(err.contains("vocab_size"), "{err}");
}

/// The Mamba-1 loader reads its own checkpoint and refuses another
/// family's by name.
#[test]
fn the_mamba1_loader_refuses_another_family() {
    let path = temp("m1_tag");
    let w = MambaWeights::init(&m1_cfg(), D_MODEL, 8);
    mamba_rs::mamba1::serialize::save(&path, &w, &m1_cfg(), D_MODEL).unwrap();
    let own = mamba_rs::mamba1::serialize::load(&path);
    let w2 = Mamba2Weights::init(&m2_cfg(), D_MODEL, 9);
    mamba_rs::mamba2::serialize::save(&path, &w2, &m2_cfg(), D_MODEL).unwrap();
    let other = mamba_rs::mamba1::serialize::load(&path);
    std::fs::remove_file(&path).ok();
    assert_eq!(own.unwrap().1, m1_cfg());
    let err = other.err().expect("a Mamba-2 file read as Mamba-1");
    assert!(err.contains("not a Mamba-1 checkpoint"), "{err}");
}
