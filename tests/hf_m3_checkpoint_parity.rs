//! The official Mamba-3 SISO checkpoints against the float64 oracle.
//!
//! `tools/oracles/m3_oracle.py` runs a checkpoint in float64 with numpy,
//! transcribed from the reference (state-spaces/mamba e9594ce), and writes
//! the logits of every prompt position and a greedy continuation. This
//! test loads the same checkpoint through `hf::load_mamba3` and holds the
//! CPU decode and prefill to those goldens.
//!
//! `#[ignore]`: it needs the downloaded checkpoint and its goldens. With
//! the default layout under `$HOME/.cache/mamba3-official` (or the
//! directory `MAMBA3_OFFICIAL_DIR` names):
//!
//!   mamba3-siso-187m/{config.json,pytorch_model.bin}
//!   goldens/siso-187m/{golden.json,last_logits.f64}
//!
//!   cargo test --release --features hf --test hf_m3_checkpoint_parity \
//!       -- --ignored --nocapture
#![cfg(feature = "hf")]

use std::path::PathBuf;

use mamba_rs::hf::embed::lm_head_logits;
use mamba_rs::hf::load_mamba3::load_hf_mamba3;
use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
use mamba_rs::mamba3_siso::cpu::prefill::{Mamba3PrefillScratch, forward_mamba3_backbone_prefill};
use mamba_rs::mamba3_siso::state::Mamba3State;
use mamba_rs::mamba3_siso::{Mamba3StepScratch, mamba3_step};

fn official_dir() -> PathBuf {
    std::env::var_os("MAMBA3_OFFICIAL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/mamba3-official")
        })
}

struct Golden {
    ids: Vec<u32>,
    logsumexp: Vec<f64>,
    target_logprob: Vec<f64>,
    top_ids: Vec<Vec<u32>>,
    last_rows_from: usize,
    last_logits: Vec<f64>,
    greedy: Vec<u32>,
    greedy_margin: Vec<f64>,
}

fn load_golden(dir: &std::path::Path) -> Golden {
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let u32s = |k: &str| -> Vec<u32> {
        json[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    };
    let f64s = |k: &str| -> Vec<f64> {
        json[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    let raw = std::fs::read(dir.join("last_logits.f64")).unwrap();
    Golden {
        ids: u32s("ids"),
        logsumexp: f64s("logsumexp"),
        target_logprob: f64s("target_logprob"),
        top_ids: json["top_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect()
            })
            .collect(),
        last_rows_from: json["last_rows_from"].as_u64().unwrap() as usize,
        last_logits: raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect(),
        greedy: u32s("greedy"),
        greedy_margin: f64s("greedy_margin"),
    }
}

fn logsumexp(v: &[f32]) -> f64 {
    let m = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    m + v
        .iter()
        .map(|&x| (f64::from(x) - m).exp())
        .sum::<f64>()
        .ln()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0 as u32
}

/// The CPU decode, token by token from the embedding, against the oracle:
/// the full logits of every recorded row, every position's log-normalizer
/// and target log-probability, the top-1 token, and the greedy
/// continuation.
#[test]
#[ignore = "needs the official checkpoint and its goldens (see the file header)"]
fn siso_187m_cpu_decode_matches_the_f64_oracle() {
    let root = official_dir();
    let golden = load_golden(&root.join("goldens/siso-187m"));
    let model = load_hf_mamba3(&root.join("mamba3-siso-187m")).unwrap();
    let cfg = model.cfg;
    let d = cfg.d_model;
    let vocab = model.vocab_size;
    let vocab_padded = model.embed.len() / d;
    assert_eq!(cfg.headdim, 64);
    assert_eq!(cfg.d_state, 128);
    assert_eq!(cfg.mlp_hidden(), 1536);

    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3StepScratch::new(&cfg);
    let mut hidden = vec![0f32; d];
    let mut logits = vec![0f32; vocab];
    let mut head_scratch = vec![0f32; vocab_padded];
    let step = |token: u32,
                state: &mut Mamba3State,
                scratch: &mut Mamba3StepScratch,
                hidden: &mut [f32],
                logits: &mut [f32],
                head_scratch: &mut [f32]| {
        let row = &model.embed[token as usize * d..(token as usize + 1) * d];
        mamba3_step(
            hidden,
            row,
            scratch,
            &model.weights,
            &mut state.layers,
            &cfg,
        );
        lm_head_logits(
            logits,
            hidden,
            &model.embed,
            vocab,
            vocab_padded,
            d,
            head_scratch,
        );
    };

    let (mut worst_rel, mut worst_lse, mut worst_lp) = (0f64, 0f64, 0f64);
    for (t, &token) in golden.ids.iter().enumerate() {
        step(
            token,
            &mut state,
            &mut scratch,
            &mut hidden,
            &mut logits,
            &mut head_scratch,
        );
        worst_lse = worst_lse.max((logsumexp(&logits) - golden.logsumexp[t]).abs());
        if let Some(&next) = golden.ids.get(t + 1) {
            let lp = f64::from(logits[next as usize]) - logsumexp(&logits);
            worst_lp = worst_lp.max((lp - golden.target_logprob[t]).abs());
        }
        let top = golden.top_ids[t][0];
        assert_eq!(argmax(&logits), top, "top-1 at position {t}");
        if t >= golden.last_rows_from {
            let r = t - golden.last_rows_from;
            let want = &golden.last_logits[r * vocab..(r + 1) * vocab];
            let scale = want.iter().fold(0f64, |a, &w| a.max(w.abs()));
            let diff = logits
                .iter()
                .zip(want)
                .fold(0f64, |a, (&g, &w)| a.max((f64::from(g) - w).abs()));
            worst_rel = worst_rel.max(diff / scale);
        }
    }
    eprintln!(
        "siso-187m CPU f32 vs f64: max|dlogit|/max|logit| = {worst_rel:.3e}, \
         max|dlse| = {worst_lse:.3e}, max|dlogprob| = {worst_lp:.3e}"
    );
    assert!(worst_rel < 1e-4, "logits rel {worst_rel:.3e}");
    assert!(worst_lp < 1e-3, "log-prob {worst_lp:.3e}");

    // Greedy continuation from the prompt's last position.
    let mut next = argmax(&logits);
    for (i, &want) in golden.greedy.iter().enumerate() {
        if golden.greedy_margin[i] >= 1e-3 {
            assert_eq!(next, want, "greedy token {i}");
        } else if next != want {
            eprintln!("greedy token {i}: tie within the oracle's margin, stopping");
            break;
        }
        step(
            next,
            &mut state,
            &mut scratch,
            &mut hidden,
            &mut logits,
            &mut head_scratch,
        );
        next = argmax(&logits);
    }
}

/// The CPU prefill of the prompt window agrees with the step chain at the
/// last position, so a prompt can be prefilled and then decoded.
#[test]
#[ignore = "needs the official checkpoint and its goldens (see the file header)"]
fn siso_187m_cpu_prefill_matches_the_f64_oracle() {
    let root = official_dir();
    let golden = load_golden(&root.join("goldens/siso-187m"));
    let model = load_hf_mamba3(&root.join("mamba3-siso-187m")).unwrap();
    let cfg = model.cfg;
    let d = cfg.d_model;
    let vocab = model.vocab_size;
    let vocab_padded = model.embed.len() / d;
    let t_len = golden.ids.len();
    let input: Vec<f32> = golden
        .ids
        .iter()
        .flat_map(|&tok| {
            model.embed[tok as usize * d..(tok as usize + 1) * d]
                .iter()
                .copied()
        })
        .collect();
    let dims = Mamba3Dims::from_config(&cfg, t_len);
    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3PrefillScratch::new(&dims, d);
    let mut rows = vec![0f32; t_len * d];
    forward_mamba3_backbone_prefill(
        &mut rows,
        &input,
        &model.weights,
        &mut state.layers,
        &mut scratch,
        &dims,
    );
    let mut logits = vec![0f32; vocab];
    let mut head_scratch = vec![0f32; vocab_padded];
    let last = t_len - 1;
    lm_head_logits(
        &mut logits,
        &rows[last * d..],
        &model.embed,
        vocab,
        vocab_padded,
        d,
        &mut head_scratch,
    );
    let r = last - golden.last_rows_from;
    let want = &golden.last_logits[r * vocab..(r + 1) * vocab];
    let scale = want.iter().fold(0f64, |a, &w| a.max(w.abs()));
    let diff = logits
        .iter()
        .zip(want)
        .fold(0f64, |a, (&g, &w)| a.max((f64::from(g) - w).abs()));
    eprintln!(
        "siso-187m CPU prefill last row: max|dlogit|/max|logit| = {:.3e}",
        diff / scale
    );
    assert!(diff / scale < 1e-4);
    assert_eq!(argmax(&logits), golden.top_ids[last][0]);
}
