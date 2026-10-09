//! The official Mamba-3 MIMO checkpoints against the float64 oracle.
//!
//! `tools/oracles/m3_oracle.py` runs a MIMO checkpoint in float64 with
//! numpy, transcribed from the reference's MIMO step and module (state-
//! spaces/mamba e9594ce), and writes the logits of every prompt position
//! and a greedy continuation. This test loads the same checkpoint through
//! `hf::load_mamba3::load_hf_mamba3_mimo` and holds the CPU decode to those
//! goldens, and with `cuda` the GPU engine in f32 and bf16.
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

/// The GPU engine on the oracle's prompt, one window then the greedy
/// continuation decoded: in f32 the recorded logits rows and every top-1;
/// in bf16 the mean prompt NLL within a few hundredths of a nat; in both
/// the greedy tokens wherever the oracle's margin clears the lane's noise.
#[cfg(feature = "cuda")]
fn mimo_187m_gpu_arm(dtype: mamba_rs::mamba_ssm::gpu::dtype::WeightDtype) {
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::mamba3_mimo::gpu::Mamba3MimoGpuEngine;

    let root = official_dir("MAMBA3_OFFICIAL_DIR", ".cache/mamba3-official");
    let golden = load_golden(&root.join("goldens/mimo-187m"));
    let model = load_hf_mamba3_mimo(&root.join("mamba3-mimo-187m")).unwrap();
    let cfg = model.cfg;
    let d = cfg.d_model;
    let vocab = model.vocab_size;
    let vocab_padded = model.embed.len() / d;
    let embed = |tok: u32| &model.embed[tok as usize * d..(tok as usize + 1) * d];
    let device = GpuDevice::new(0).unwrap();
    let engine = Mamba3MimoGpuEngine::new(&device, &model.weights, (cfg, d, 1), dtype).unwrap();
    let mut state = engine.alloc_state().unwrap();
    let t_len = golden.ids.len();
    let mut win = engine.alloc_window(t_len).unwrap();
    let prompt: Vec<f32> = golden
        .ids
        .iter()
        .flat_map(|&tok| embed(tok).iter().copied())
        .collect();
    let mut rows = vec![0f32; t_len * d];
    engine
        .run(&prompt, &mut rows, &mut state, &mut win)
        .unwrap();
    let mut logits = vec![0f32; vocab];
    let mut head_scratch = vec![0f32; vocab_padded];
    let (mut worst, mut nll, mut nll_oracle) = (0f64, 0f64, 0f64);
    for t in 0..t_len {
        lm_head_logits(
            &mut logits,
            &rows[t * d..(t + 1) * d],
            &model.embed,
            vocab,
            vocab_padded,
            d,
            &mut head_scratch,
        );
        if let Some(&next) = golden.ids.get(t + 1) {
            nll -= f64::from(logits[next as usize]) - logsumexp(&logits);
            nll_oracle -= golden.target_logprob[t];
        }
        if dtype == WeightDtype::F32 {
            assert_eq!(argmax(&logits), golden.top_ids[t][0], "top-1 at {t}");
        }
        if t >= golden.last_rows_from {
            let r = t - golden.last_rows_from;
            let want = &golden.last_logits[r * vocab..(r + 1) * vocab];
            let scale = want.iter().fold(0f64, |a, &w| a.max(w.abs()));
            let diff = logits
                .iter()
                .zip(want)
                .fold(0f64, |a, (&g, &w)| a.max((f64::from(g) - w).abs()));
            worst = worst.max(diff / scale);
        }
    }
    let n = (t_len - 1) as f64;
    let (nll, nll_oracle) = (nll / n, nll_oracle / n);
    eprintln!(
        "mimo-187m GPU {dtype:?} window: max|dlogit|/max|logit| = {worst:.3e}, \
         mean NLL {nll:.4} vs oracle {nll_oracle:.4}"
    );
    let margin = if dtype == WeightDtype::F32 {
        assert!(worst < 2e-3, "GPU logits rel {worst:.3e}");
        1e-2
    } else {
        assert!(
            (nll - nll_oracle).abs() < 3e-2,
            "{dtype:?} NLL {nll:.4} vs {nll_oracle:.4}"
        );
        0.25
    };
    let mut one = engine.alloc_window(1).unwrap();
    let mut hidden = vec![0f32; d];
    let mut next = argmax(&logits);
    let mut produced = 0usize;
    for (i, &want) in golden.greedy.iter().enumerate() {
        if next != want {
            assert!(
                golden.greedy_margin[i] < margin,
                "greedy token {i}: {next} vs {want}"
            );
            eprintln!("greedy token {i}: within the lane's margin, stopping");
            break;
        }
        produced += 1;
        engine
            .run(embed(next), &mut hidden, &mut state, &mut one)
            .unwrap();
        lm_head_logits(
            &mut logits,
            &hidden,
            &model.embed,
            vocab,
            vocab_padded,
            d,
            &mut head_scratch,
        );
        next = argmax(&logits);
    }
    eprintln!("mimo-187m GPU {dtype:?} greedy: {produced} tokens match the oracle");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the official checkpoint, its goldens and a CUDA device"]
fn mimo_187m_gpu_matches_the_f64_oracle() {
    mimo_187m_gpu_arm(mamba_rs::mamba_ssm::gpu::dtype::WeightDtype::F32);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the official checkpoint, its goldens and a CUDA device"]
fn mimo_187m_gpu_bf16_tracks_the_f64_oracle() {
    mimo_187m_gpu_arm(mamba_rs::mamba_ssm::gpu::dtype::WeightDtype::Bf16);
}
