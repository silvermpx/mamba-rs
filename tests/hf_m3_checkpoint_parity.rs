//! The official Mamba-3 SISO checkpoints against the float64 oracle.
//!
//! `tools/oracles/m3_oracle.py` runs a checkpoint in float64 with numpy,
//! transcribed from the reference (state-spaces/mamba e9594ce), and writes
//! the logits of every prompt position and a greedy continuation. This
//! test loads the same checkpoint through `hf::load_mamba3` and holds the
//! CPU decode and prefill to those goldens, and with `cuda` the f32 and bf16
//! GPU engines too.
//!
//! `#[ignore]`: it needs the downloaded checkpoint and its goldens. With
//! the default layout under `$HOME/.cache/mamba3-official` (or the
//! directory `MAMBA3_OFFICIAL_DIR` names):
//!
//!   mamba3-siso-187m/{config.json,pytorch_model.bin}
//!   goldens/siso-187m/{golden.json,last_logits.f64}
//!
//!   cargo test --release --features hf[,cuda] --test hf_m3_checkpoint_parity \
//!       -- --ignored --nocapture
#![cfg(feature = "hf")]

use mamba_rs::hf::embed::lm_head_logits;
use mamba_rs::hf::load_mamba3;
use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
use mamba_rs::mamba3_siso::cpu::inference::{Mamba3StepScratch, mamba3_step};
use mamba_rs::mamba3_siso::cpu::prefill::{Mamba3PrefillScratch, forward_mamba3_backbone_prefill};
use mamba_rs::mamba3_siso::state::Mamba3State;

#[path = "common/m3_golden.rs"]
mod m3_golden;
use m3_golden::{argmax, load_golden, logsumexp, official_dir};

/// The CPU decode, token by token from the embedding, against the oracle:
/// the full logits of every recorded row, every position's log-normalizer
/// and target log-probability, the top-1 token, and the greedy
/// continuation.
#[test]
#[ignore = "needs the official checkpoint and its goldens (see the file header)"]
fn siso_187m_cpu_decode_matches_the_f64_oracle() {
    let root = official_dir("MAMBA3_OFFICIAL_DIR", ".cache/mamba3-official");
    let golden = load_golden(&root.join("goldens/siso-187m"));
    let model = load_mamba3(&root.join("mamba3-siso-187m")).unwrap();
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
    let root = official_dir("MAMBA3_OFFICIAL_DIR", ".cache/mamba3-official");
    let golden = load_golden(&root.join("goldens/siso-187m"));
    let model = load_mamba3(&root.join("mamba3-siso-187m")).unwrap();
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

// GPU against the float64 oracle is a tolerance comparison: 2e-3 relative
// on the logits is the sm_120 floor the chunked prefill parity measured,
// with headroom, and the top-1 checks excuse positions whose oracle margin
// is inside that noise.
#[cfg(feature = "cuda")]
const GPU_REL_TOL: f64 = 2e-3;
#[cfg(feature = "cuda")]
const GPU_MARGIN: f64 = 1e-2;

#[cfg(feature = "cuda")]
fn top_margin(golden: &m3_golden::Golden, t: usize) -> f64 {
    let json_top = &golden.top_ids[t];
    // The oracle keeps full logits only for its last rows; elsewhere the
    // margin is unknown and the top-1 check stays strict.
    if t >= golden.last_rows_from {
        let vocab = golden.last_logits.len() / (golden.ids.len() - golden.last_rows_from);
        let r = t - golden.last_rows_from;
        let row = &golden.last_logits[r * vocab..(r + 1) * vocab];
        row[json_top[0] as usize] - row[json_top[1] as usize]
    } else {
        f64::INFINITY
    }
}

/// The f32 GPU engine (block-tier decode step at headdim 64, the gated MLP
/// on the GPU) token by token against the oracle, then its greedy
/// continuation.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the official checkpoint, its goldens and a CUDA device"]
fn siso_187m_gpu_decode_matches_the_f64_oracle() {
    use mamba_rs::gpu::device::GpuDevice;
    use mamba_rs::mamba3_siso::gpu::inference::Mamba3GpuEngine;

    let root = official_dir("MAMBA3_OFFICIAL_DIR", ".cache/mamba3-official");
    let golden = load_golden(&root.join("goldens/siso-187m"));
    let model = load_mamba3(&root.join("mamba3-siso-187m")).unwrap();
    let cfg = model.cfg;
    let d = cfg.d_model;
    let vocab = model.vocab_size;
    let vocab_padded = model.embed.len() / d;
    let device = GpuDevice::new(0).unwrap();
    let engine = Mamba3GpuEngine::new(&device, &model.weights, cfg, d, 1).unwrap();
    let mut state = engine.alloc_state().unwrap();
    let mut scratch = engine.alloc_scratch().unwrap();
    let mut hidden = vec![0f32; d];
    let mut logits = vec![0f32; vocab];
    let mut head_scratch = vec![0f32; vocab_padded];
    let mut worst = 0f64;
    for (t, &token) in golden.ids.iter().enumerate() {
        let row = &model.embed[token as usize * d..(token as usize + 1) * d];
        engine
            .step(row, &mut hidden, &mut state, &mut scratch)
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
        if top_margin(&golden, t) >= GPU_MARGIN {
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
    eprintln!("siso-187m GPU f32 decode vs f64: max|dlogit|/max|logit| = {worst:.3e}");
    assert!(worst < GPU_REL_TOL, "GPU logits rel {worst:.3e}");
    let mut next = argmax(&logits);
    let mut produced = Vec::new();
    for (i, &want) in golden.greedy.iter().enumerate() {
        if golden.greedy_margin[i] < GPU_MARGIN && next != want {
            eprintln!("greedy token {i}: within the margin, stopping");
            break;
        }
        assert_eq!(next, want, "greedy token {i}");
        produced.push(next);
        let row = &model.embed[next as usize * d..(next as usize + 1) * d];
        engine
            .step(row, &mut hidden, &mut state, &mut scratch)
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
    eprintln!(
        "siso-187m GPU greedy: {} tokens match the oracle",
        produced.len()
    );
}

/// The prompt prefilled on the GPU in one chunked pass (MLP rows included),
/// then decoded: the last prompt position and the greedy continuation
/// against the oracle.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the official checkpoint, its goldens and a CUDA device"]
fn siso_187m_gpu_prefill_then_decode_matches_the_f64_oracle() {
    use mamba_rs::gpu::buffers::GpuBuffer;
    use mamba_rs::gpu::device::GpuDevice;
    use mamba_rs::mamba3_siso::gpu::inference::Mamba3GpuEngine;

    let root = official_dir("MAMBA3_OFFICIAL_DIR", ".cache/mamba3-official");
    let golden = load_golden(&root.join("goldens/siso-187m"));
    let model = load_mamba3(&root.join("mamba3-siso-187m")).unwrap();
    let cfg = model.cfg;
    let d = cfg.d_model;
    let vocab = model.vocab_size;
    let vocab_padded = model.embed.len() / d;
    let t_len = golden.ids.len();
    let device = GpuDevice::new(0).unwrap();
    let engine = Mamba3GpuEngine::new(&device, &model.weights, cfg, d, 1).unwrap();
    let mut state = engine.alloc_state().unwrap();
    let mut scratch = engine.alloc_scratch().unwrap();
    let mut prefill = engine.alloc_prefill(t_len).unwrap();
    let prompt: Vec<f32> = golden
        .ids
        .iter()
        .flat_map(|&tok| {
            model.embed[tok as usize * d..(tok as usize + 1) * d]
                .iter()
                .copied()
        })
        .collect();
    let stream = &engine.ctx().stream;
    let gpu_prompt = GpuBuffer::from_cpu(stream, &prompt).unwrap();
    let mut last_hidden = GpuBuffer::zeros(stream, d).unwrap();
    engine
        .prefill_sequence(
            &mut prefill,
            &gpu_prompt,
            t_len,
            &mut state,
            false,
            &mut last_hidden,
        )
        .unwrap();
    stream.synchronize().unwrap();
    let mut hidden = vec![0f32; d];
    last_hidden.download(stream, &mut hidden).unwrap();
    let mut logits = vec![0f32; vocab];
    let mut head_scratch = vec![0f32; vocab_padded];
    lm_head_logits(
        &mut logits,
        &hidden,
        &model.embed,
        vocab,
        vocab_padded,
        d,
        &mut head_scratch,
    );
    let r = t_len - 1 - golden.last_rows_from;
    let want = &golden.last_logits[r * vocab..(r + 1) * vocab];
    let scale = want.iter().fold(0f64, |a, &w| a.max(w.abs()));
    let diff = logits
        .iter()
        .zip(want)
        .fold(0f64, |a, (&g, &w)| a.max((f64::from(g) - w).abs()));
    eprintln!(
        "siso-187m GPU prefill last row: max|dlogit|/max|logit| = {:.3e}",
        diff / scale
    );
    assert!(diff / scale < GPU_REL_TOL);
    let mut next = argmax(&logits);
    for (i, &want) in golden.greedy.iter().enumerate() {
        if golden.greedy_margin[i] < GPU_MARGIN && next != want {
            eprintln!("greedy token {i}: within the margin, stopping");
            break;
        }
        assert_eq!(next, want, "greedy token {i} after prefill");
        let row = &model.embed[next as usize * d..(next as usize + 1) * d];
        engine
            .step(row, &mut hidden, &mut state, &mut scratch)
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
}

// bf16 storage and activations against the float64 oracle: a bf16
// inference of a real LM moves its mean prompt NLL by hundredths of a nat,
// and its top-1 only where the oracle's own margin is a fraction of a nat.
#[cfg(feature = "cuda")]
const BF16_NLL_TOL: f64 = 3e-2;
#[cfg(feature = "cuda")]
const BF16_MARGIN: f64 = 0.25;

/// The bf16 engine, decode token by token and then a typed prefill of the
/// same prompt, against the oracle: the mean prompt NLL, the top-1 where the
/// oracle's margin clears bf16 noise, and the greedy continuation.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the official checkpoint, its goldens and a CUDA device"]
fn siso_187m_gpu_bf16_tracks_the_f64_oracle() {
    use mamba_rs::gpu::buffers::GpuBuffer;
    use mamba_rs::gpu::device::GpuDevice;
    use mamba_rs::gpu::dtype::WeightDtype;
    use mamba_rs::mamba3_siso::gpu::inference::Mamba3GpuEngineMixed;

    let root = official_dir("MAMBA3_OFFICIAL_DIR", ".cache/mamba3-official");
    let golden = load_golden(&root.join("goldens/siso-187m"));
    let model = load_mamba3(&root.join("mamba3-siso-187m")).unwrap();
    let cfg = model.cfg;
    let d = cfg.d_model;
    let vocab = model.vocab_size;
    let vocab_padded = model.embed.len() / d;
    let t_len = golden.ids.len();
    let device = GpuDevice::new(0).unwrap();
    let engine =
        Mamba3GpuEngineMixed::new(&device, &model.weights, cfg, d, 1, WeightDtype::Bf16).unwrap();
    let mut hidden = vec![0f32; d];
    let mut logits = vec![0f32; vocab];
    let mut head_scratch = vec![0f32; vocab_padded];
    let embed = |tok: u32| &model.embed[tok as usize * d..(tok as usize + 1) * d];

    let mut state = engine.alloc_state().unwrap();
    let mut scratch = engine.alloc_mixed_scratch().unwrap();
    let (mut nll, mut nll_oracle) = (0f64, 0f64);
    for (t, &token) in golden.ids.iter().enumerate() {
        engine
            .step_mixed_native(embed(token), &mut hidden, &mut state, &mut scratch)
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
        if let Some(&next) = golden.ids.get(t + 1) {
            nll -= f64::from(logits[next as usize]) - logsumexp(&logits);
            nll_oracle -= golden.target_logprob[t];
        }
        if t >= golden.last_rows_from && top_margin(&golden, t) >= BF16_MARGIN {
            assert_eq!(argmax(&logits), golden.top_ids[t][0], "top-1 at {t}");
        }
    }
    let n = (t_len - 1) as f64;
    let (nll, nll_oracle) = (nll / n, nll_oracle / n);
    eprintln!("siso-187m GPU bf16 decode: mean NLL {nll:.4} vs oracle {nll_oracle:.4}");
    assert!(
        (nll - nll_oracle).abs() < BF16_NLL_TOL,
        "bf16 mean NLL {nll:.4} vs {nll_oracle:.4}"
    );

    // The typed prefill of the same prompt, then the greedy continuation.
    let mut state = engine.alloc_state().unwrap();
    let mut prefill = engine.alloc_prefill(t_len).unwrap();
    let prompt: Vec<f32> = golden
        .ids
        .iter()
        .flat_map(|&tok| embed(tok).iter().copied())
        .collect();
    let stream = engine.ctx_stream();
    let gpu_prompt = GpuBuffer::from_cpu(stream, &prompt).unwrap();
    let mut last_hidden = GpuBuffer::zeros(stream, d).unwrap();
    engine
        .prefill_sequence(
            &mut prefill,
            &gpu_prompt,
            t_len,
            &mut state,
            false,
            &mut last_hidden,
        )
        .unwrap();
    stream.synchronize().unwrap();
    last_hidden.download(stream, &mut hidden).unwrap();
    stream.synchronize().unwrap();
    lm_head_logits(
        &mut logits,
        &hidden,
        &model.embed,
        vocab,
        vocab_padded,
        d,
        &mut head_scratch,
    );
    let mut next = argmax(&logits);
    let mut produced = 0usize;
    for (i, &want) in golden.greedy.iter().enumerate() {
        if next != want {
            assert!(
                golden.greedy_margin[i] < BF16_MARGIN,
                "greedy token {i}: {next} vs {want} at margin {}",
                golden.greedy_margin[i]
            );
            eprintln!("greedy token {i}: within the bf16 margin, stopping");
            break;
        }
        produced += 1;
        engine
            .step_mixed_native(embed(next), &mut hidden, &mut state, &mut scratch)
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
    eprintln!("siso-187m GPU bf16 prefill + greedy: {produced} tokens match the oracle");
}
