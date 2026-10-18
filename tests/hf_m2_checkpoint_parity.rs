//! The official Mamba-2 checkpoints against the float64 oracle.
//!
//! `tools/oracles/m2_oracle.py` runs a Mamba-2 checkpoint in float64 with
//! numpy, transcribed from the reference's step (state-spaces/mamba
//! e9594ce), and writes the logits of every prompt position and a greedy
//! continuation. This test loads the same checkpoint through
//! `hf::load_mamba2` and holds the CPU decode and prefill to those goldens,
//! and with `cuda` the GPU prefill and decode.
//!
//! `#[ignore]`: it needs the downloaded checkpoint and its goldens, under
//! `$HOME/.cache/mamba2-official` or the directory `MAMBA2_OFFICIAL_DIR`
//! names:
//!
//!   mamba2-130m/{config.json,pytorch_model.bin}
//!   goldens/mamba2-130m/{golden.json,last_logits.f64}
//!
//! and, for the transformers layout, `mamba2-130m-hf/{config.json,
//! model.safetensors}` from AntonV/mamba2-130m-hf.
//!
//!   cargo test --release --features hf --test hf_m2_checkpoint_parity \
//!       -- --ignored --nocapture
#![cfg(feature = "hf")]

use mamba_rs::hf::embed::lm_head_logits;
use mamba_rs::hf::load_mamba2;
use mamba_rs::mamba2::Mamba2State;
use mamba_rs::mamba2::cpu::{
    Mamba2PrefillScratch, Mamba2StepScratch, forward_mamba2_backbone_prefill, mamba2_step,
};

#[path = "common/m3_golden.rs"]
mod m3_golden;
use m3_golden::{argmax, load_golden, logsumexp, official_dir};

/// The CPU decode token by token from the embedding, against the oracle:
/// the full logits of the recorded rows, every log-normalizer and target
/// log-probability, every top-1 and the greedy continuation; then the
/// prompt prefilled in one pass, whose last row is the decode's.
/// The transformers-layout conversion loads the model the original
/// checkpoint loads, bit for bit. Its config states the padded vocabulary
/// as the vocabulary, so only the vocabulary sizes differ.
#[test]
#[ignore = "needs both checkpoint layouts (see the file header)"]
fn mamba2_130m_transformers_layout_is_the_original() {
    let root = official_dir("MAMBA2_OFFICIAL_DIR", ".cache/mamba2-official");
    let a = load_mamba2(&root.join("mamba2-130m")).unwrap();
    let b = load_mamba2(&root.join("mamba2-130m-hf")).unwrap();
    assert_eq!(a.cfg, b.cfg);
    assert_eq!((a.vocab_size, b.vocab_size), (50277, 50288));
    assert!(a.lm_head.is_none() && b.lm_head.is_none());
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert!(bits(&a.embed) == bits(&b.embed), "embedding");
    assert!(bits(&a.weights.norm_f_weight) == bits(&b.weights.norm_f_weight));
    for (i, (la, lb)) in a.weights.layers.iter().zip(&b.weights.layers).enumerate() {
        for (name, x, y) in [
            ("norm", &la.norm_weight, &lb.norm_weight),
            ("in_proj", &la.in_proj_w, &lb.in_proj_w),
            ("conv", &la.conv_w, &lb.conv_w),
            ("conv bias", &la.conv_b, &lb.conv_b),
            ("dt bias", &la.dt_bias, &lb.dt_bias),
            ("A_log", &la.a_log, &lb.a_log),
            ("D", &la.d_param, &lb.d_param),
            ("gated norm", &la.gate_norm_weight, &lb.gate_norm_weight),
            ("out_proj", &la.out_proj_w, &lb.out_proj_w),
        ] {
            assert!(bits(x) == bits(y), "layer {i} {name}");
        }
    }
}

#[test]
#[ignore = "needs the official checkpoint and its goldens (see the file header)"]
fn mamba2_130m_cpu_matches_the_f64_oracle() {
    let root = official_dir("MAMBA2_OFFICIAL_DIR", ".cache/mamba2-official");
    let golden = load_golden(&root.join("goldens/mamba2-130m"));
    let model = load_mamba2(&root.join("mamba2-130m")).unwrap();
    let cfg = model.cfg;
    let d = cfg.d_model;
    let vocab = model.vocab_size;
    let vocab_padded = model.embed.len() / d;
    let embed = |tok: u32| &model.embed[tok as usize * d..(tok as usize + 1) * d];

    let mut state = Mamba2State::zeros(&cfg);
    let mut scratch = Mamba2StepScratch::new(&cfg);
    let mut hidden = vec![0f32; d];
    let mut logits = vec![0f32; vocab];
    let mut head_scratch = vec![0f32; vocab_padded];
    let mut step = |token: u32, hidden: &mut [f32], logits: &mut [f32]| {
        mamba2_step(
            hidden,
            embed(token),
            &mut scratch,
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
            &mut head_scratch,
        );
    };

    let (mut worst_rel, mut worst_lse, mut worst_lp) = (0f64, 0f64, 0f64);
    for (t, &token) in golden.ids.iter().enumerate() {
        step(token, &mut hidden, &mut logits);
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
    let last_hidden = hidden.clone();
    eprintln!(
        "mamba2-130m CPU f32 vs f64: max|dlogit|/max|logit| = {worst_rel:.3e}, \
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
        step(next, &mut hidden, &mut logits);
        next = argmax(&logits);
    }
    eprintln!("mamba2-130m CPU greedy: {produced} tokens match the oracle");

    // The prompt in one pass: its last row is the decode chain's.
    let t_len = golden.ids.len();
    let prompt: Vec<f32> = golden
        .ids
        .iter()
        .flat_map(|&tok| embed(tok).iter().copied())
        .collect();
    let mut st = Mamba2State::zeros(&cfg);
    let mut ps = Mamba2PrefillScratch::new();
    let mut rows = vec![0f32; t_len * d];
    forward_mamba2_backbone_prefill(
        &mut rows,
        &prompt,
        &model.weights,
        &mut st.layers,
        &mut ps,
        (&cfg, t_len),
    );
    let last = &rows[(t_len - 1) * d..];
    if cfg!(any(feature = "accelerate", feature = "gemm-blas")) {
        let rel = last
            .iter()
            .zip(&last_hidden)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(rel < 1e-4, "prefill last row {rel:.3e}");
    } else {
        assert_eq!(last, &last_hidden[..], "prefill last row bits");
    }
}

/// The f32 GPU engine against the oracle: the prompt prefilled in one
/// chunked pass, then the greedy continuation decoded, both on the GPU.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the official checkpoint, its goldens and a CUDA device"]
fn mamba2_130m_gpu_matches_the_f64_oracle() {
    use mamba_rs::gpu::WeightDtype;
    use mamba_rs::gpu::device::GpuDevice;
    use mamba_rs::mamba2::gpu::Mamba2GpuEngine;

    let root = official_dir("MAMBA2_OFFICIAL_DIR", ".cache/mamba2-official");
    let golden = load_golden(&root.join("goldens/mamba2-130m"));
    let model = load_mamba2(&root.join("mamba2-130m")).unwrap();
    let cfg = model.cfg;
    let d = cfg.d_model;
    let vocab = model.vocab_size;
    let vocab_padded = model.embed.len() / d;
    let embed = |tok: u32| &model.embed[tok as usize * d..(tok as usize + 1) * d];
    let device = GpuDevice::new(0).unwrap();
    let engine =
        Mamba2GpuEngine::new(&device, &model.weights, cfg, d, 1, WeightDtype::F32).unwrap();
    let mut state = engine.alloc_state().unwrap();
    let mut scratch = engine.alloc_scratch().unwrap();
    let t_len = golden.ids.len();
    let mut pf = engine.alloc_prefill(t_len).unwrap();
    let prompt: Vec<f32> = golden
        .ids
        .iter()
        .flat_map(|&tok| embed(tok).iter().copied())
        .collect();
    let mut rows = vec![0f32; t_len * d];
    engine
        .prefill(&prompt, &mut rows, &mut state, &mut pf)
        .unwrap();
    let mut logits = vec![0f32; vocab];
    let mut head_scratch = vec![0f32; vocab_padded];
    let mut worst = 0f64;
    for t in golden.last_rows_from..t_len {
        lm_head_logits(
            &mut logits,
            &rows[t * d..(t + 1) * d],
            &model.embed,
            vocab,
            vocab_padded,
            d,
            &mut head_scratch,
        );
        let r = t - golden.last_rows_from;
        let want = &golden.last_logits[r * vocab..(r + 1) * vocab];
        let scale = want.iter().fold(0f64, |a, &w| a.max(w.abs()));
        let diff = logits
            .iter()
            .zip(want)
            .fold(0f64, |a, (&g, &w)| a.max((f64::from(g) - w).abs()));
        worst = worst.max(diff / scale);
        assert_eq!(argmax(&logits), golden.top_ids[t][0], "top-1 at {t}");
    }
    eprintln!("mamba2-130m GPU prefill vs f64: max|dlogit|/max|logit| = {worst:.3e}");
    assert!(worst < 2e-3, "GPU logits rel {worst:.3e}");
    let mut hidden = vec![0f32; d];
    let mut next = argmax(&logits);
    let mut produced = 0usize;
    for (i, &want) in golden.greedy.iter().enumerate() {
        if next != want {
            assert!(
                golden.greedy_margin[i] < 1e-2,
                "greedy token {i}: {next} vs {want}"
            );
            eprintln!("greedy token {i}: within the margin, stopping");
            break;
        }
        produced += 1;
        engine
            .step(embed(next), &mut hidden, &mut state, &mut scratch)
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
    eprintln!("mamba2-130m GPU greedy: {produced} tokens match the oracle");
}

/// The bf16 engine on the oracle's prompt: the mean prompt NLL within a
/// few hundredths of a nat and the greedy continuation where the oracle's
/// margin clears bf16 noise.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the official checkpoint, its goldens and a CUDA device"]
fn mamba2_130m_gpu_bf16_tracks_the_f64_oracle() {
    use mamba_rs::gpu::device::GpuDevice;
    use mamba_rs::gpu::dtype::WeightDtype;
    use mamba_rs::mamba2::gpu::Mamba2GpuEngine;

    let root = official_dir("MAMBA2_OFFICIAL_DIR", ".cache/mamba2-official");
    let golden = load_golden(&root.join("goldens/mamba2-130m"));
    let model = load_mamba2(&root.join("mamba2-130m")).unwrap();
    let cfg = model.cfg;
    let d = cfg.d_model;
    let vocab = model.vocab_size;
    let vocab_padded = model.embed.len() / d;
    let embed = |tok: u32| &model.embed[tok as usize * d..(tok as usize + 1) * d];
    let device = GpuDevice::new(0).unwrap();
    let engine =
        Mamba2GpuEngine::new(&device, &model.weights, cfg, d, 1, WeightDtype::Bf16).unwrap();
    let mut state = engine.alloc_state().unwrap();
    let mut scratch = engine.alloc_scratch().unwrap();
    let t_len = golden.ids.len();
    let mut pf = engine.alloc_prefill(t_len).unwrap();
    let prompt: Vec<f32> = golden
        .ids
        .iter()
        .flat_map(|&tok| embed(tok).iter().copied())
        .collect();
    let mut rows = vec![0f32; t_len * d];
    engine
        .prefill(&prompt, &mut rows, &mut state, &mut pf)
        .unwrap();
    let mut logits = vec![0f32; vocab];
    let mut head_scratch = vec![0f32; vocab_padded];
    let (mut nll, mut nll_oracle) = (0f64, 0f64);
    for t in 0..t_len - 1 {
        lm_head_logits(
            &mut logits,
            &rows[t * d..(t + 1) * d],
            &model.embed,
            vocab,
            vocab_padded,
            d,
            &mut head_scratch,
        );
        nll -= f64::from(logits[golden.ids[t + 1] as usize]) - logsumexp(&logits);
        nll_oracle -= golden.target_logprob[t];
    }
    let n = (t_len - 1) as f64;
    let (nll, nll_oracle) = (nll / n, nll_oracle / n);
    eprintln!("mamba2-130m GPU bf16 prefill: mean NLL {nll:.4} vs oracle {nll_oracle:.4}");
    assert!(
        (nll - nll_oracle).abs() < 3e-2,
        "bf16 NLL {nll:.4} vs {nll_oracle:.4}"
    );
    lm_head_logits(
        &mut logits,
        &rows[(t_len - 1) * d..],
        &model.embed,
        vocab,
        vocab_padded,
        d,
        &mut head_scratch,
    );
    let mut hidden = vec![0f32; d];
    let mut next = argmax(&logits);
    let mut produced = 0usize;
    for (i, &want) in golden.greedy.iter().enumerate() {
        if next != want {
            assert!(
                golden.greedy_margin[i] < 0.25,
                "greedy token {i}: {next} vs {want}"
            );
            eprintln!("greedy token {i}: within the bf16 margin, stopping");
            break;
        }
        produced += 1;
        engine
            .step(embed(next), &mut hidden, &mut state, &mut scratch)
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
    eprintln!("mamba2-130m GPU bf16 greedy: {produced} tokens match the oracle");
}
