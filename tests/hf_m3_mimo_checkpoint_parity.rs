//! The official Mamba-3 MIMO checkpoints against the float64 oracle.
//!
//! `tools/oracles/m3_oracle.py` runs a MIMO checkpoint in float64 with
//! numpy, transcribed from the reference's MIMO step and module (state-
//! spaces/mamba e9594ce), and writes the logits of every prompt position
//! and a greedy continuation. This test loads the same checkpoint through
//! `hf::load_mamba3::load_hf_mamba3_mimo` and holds the CPU decode to those
//! goldens.
//!
//! `#[ignore]`: it needs the downloaded checkpoint and its goldens, under
//! `$HOME/.cache/mamba3-official` or the directory `MAMBA3_OFFICIAL_DIR`
//! names:
//!
//!   mamba3-mimo-187m/{config.json,pytorch_model.bin}
//!   goldens/mimo-187m/{golden.json,last_logits.f64}
//!
//!   cargo test --release --features hf --test hf_m3_mimo_checkpoint_parity \
//!       -- --ignored --nocapture
#![cfg(feature = "hf")]

use mamba_rs::hf::embed::lm_head_logits;
use mamba_rs::hf::load_mamba3::load_hf_mamba3_mimo;
use mamba_rs::mamba3_mimo::{Mamba3MimoState, Mamba3MimoStepScratch, mamba3_mimo_step};

#[path = "common/m3_golden.rs"]
mod m3_golden;
use m3_golden::{argmax, load_golden, logsumexp, official_dir};

/// The CPU decode, token by token from the embedding, against the oracle:
/// the full logits of the recorded rows, every position's log-normalizer
/// and target log-probability, the top-1 token, and the greedy
/// continuation.
#[test]
#[ignore = "needs the official checkpoint and its goldens (see the file header)"]
fn mimo_187m_cpu_decode_matches_the_f64_oracle() {
    let root = official_dir("MAMBA3_OFFICIAL_DIR", ".cache/mamba3-official");
    let golden = load_golden(&root.join("goldens/mimo-187m"));
    let model = load_hf_mamba3_mimo(&root.join("mamba3-mimo-187m")).unwrap();
    let cfg = model.cfg;
    let d = cfg.d_model;
    let vocab = model.vocab_size;
    let vocab_padded = model.embed.len() / d;
    assert_eq!(cfg.mimo_rank, 4);
    assert_eq!((cfg.headdim, cfg.d_state), (64, 128));
    assert_eq!(cfg.mlp_hidden(), 1280);

    let mut state = Mamba3MimoState::zeros(&cfg);
    let mut scratch = Mamba3MimoStepScratch::new(&cfg);
    let mut hidden = vec![0f32; d];
    let mut logits = vec![0f32; vocab];
    let mut head_scratch = vec![0f32; vocab_padded];
    let mut step = |token: u32, logits: &mut [f32]| {
        let row = &model.embed[token as usize * d..(token as usize + 1) * d];
        mamba3_mimo_step(
            &mut hidden,
            row,
            &mut scratch,
            &model.weights,
            &mut state.layers,
            &cfg,
        );
        lm_head_logits(
            logits,
            &hidden,
            &model.embed,
            vocab,
            vocab_padded,
            d,
            &mut head_scratch,
        );
    };

    let (mut worst_rel, mut worst_lse, mut worst_lp) = (0f64, 0f64, 0f64);
    for (t, &token) in golden.ids.iter().enumerate() {
        step(token, &mut logits);
        worst_lse = worst_lse.max((logsumexp(&logits) - golden.logsumexp[t]).abs());
        if let Some(&next) = golden.ids.get(t + 1) {
            let lp = f64::from(logits[next as usize]) - logsumexp(&logits);
            worst_lp = worst_lp.max((lp - golden.target_logprob[t]).abs());
        }
        assert_eq!(
            argmax(&logits),
            golden.top_ids[t][0],
            "top-1 at position {t}"
        );
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
        "mimo-187m CPU f32 vs f64: max|dlogit|/max|logit| = {worst_rel:.3e}, \
         max|dlse| = {worst_lse:.3e}, max|dlogprob| = {worst_lp:.3e}"
    );
    assert!(worst_rel < 1e-4, "logits rel {worst_rel:.3e}");
    assert!(worst_lp < 1e-3, "log-prob {worst_lp:.3e}");

    let mut next = argmax(&logits);
    let mut produced = 0usize;
    for (i, &want) in golden.greedy.iter().enumerate() {
        if golden.greedy_margin[i] >= 1e-3 {
            assert_eq!(next, want, "greedy token {i}");
        } else if next != want {
            eprintln!("greedy token {i}: tie within the oracle's margin, stopping");
            break;
        }
        produced += 1;
        step(next, &mut logits);
        next = argmax(&logits);
    }
    eprintln!("mimo-187m CPU greedy: {produced} tokens match the oracle");
}
