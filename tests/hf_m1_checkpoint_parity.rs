//! The official Mamba-1 checkpoints against the float64 oracle.
//!
//! `tools/oracles/m1_oracle.py` runs a Mamba-1 checkpoint in float64 with
//! numpy, transcribed from the reference's step (state-spaces/mamba
//! e9594ce), and writes the logits of every prompt position and a greedy
//! continuation. This test loads the same checkpoint through `hf::load_hf`
//! and holds the CPU decode and prefill to those goldens, and with `cuda`
//! the GPU decode and prefill in f32 and bf16.
//!
//! `#[ignore]`: it needs the downloaded checkpoint and its goldens, under
//! `$HOME/.cache/mamba1-official` or the directory `MAMBA1_OFFICIAL_DIR`
//! names:
//!
//!   mamba-130m-hf/{config.json,model.safetensors}
//!   goldens/mamba-130m/{golden.json,last_logits.f64}
//!
//! and, for the original repository's layout, `mamba-130m/{config.json,
//! pytorch_model.bin}` from state-spaces/mamba-130m.
//!
//!   cargo test --release --features hf --test hf_m1_checkpoint_parity \
//!       -- --ignored --nocapture
#![cfg(feature = "hf")]

#[cfg(feature = "cuda")]
use mamba_rs::gpu::GpuOptions;
use mamba_rs::hf::embed::lm_head_logits;
use mamba_rs::hf::load::{HfModel, load_hf};
use mamba_rs::mamba1::cpu::prefill::PrefillMode;

#[path = "common/m3_golden.rs"]
mod m3_golden;
use m3_golden::{Golden, argmax, load_golden, logsumexp, official_dir};

fn checkpoint() -> (HfModel, Golden) {
    let root = official_dir("MAMBA1_OFFICIAL_DIR", ".cache/mamba1-official");
    let golden = load_golden(&root.join("goldens/mamba-130m"));
    let model = load_hf(&root.join("mamba-130m-hf")).unwrap();
    (model, golden)
}

/// Logits of one hidden row through the tied head.
struct Head<'a> {
    model: &'a HfModel,
    scratch: Vec<f32>,
}

impl<'a> Head<'a> {
    fn new(model: &'a HfModel) -> Self {
        Self {
            model,
            scratch: vec![0f32; model.vocab_size_padded],
        }
    }

    fn logits(&mut self, hidden: &[f32], logits: &mut [f32]) {
        let m = self.model;
        lm_head_logits(
            logits,
            hidden,
            &m.embed,
            m.vocab_size,
            m.vocab_size_padded,
            m.d_model,
            &mut self.scratch,
        );
    }
}

fn embed(model: &HfModel, token: u32) -> &[f32] {
    let d = model.d_model;
    &model.embed[token as usize * d..(token as usize + 1) * d]
}

/// The largest |logit - oracle| over the largest |oracle logit| of row `t`.
fn row_error(golden: &Golden, t: usize, vocab: usize, logits: &[f32]) -> f64 {
    let r = t - golden.last_rows_from;
    let want = &golden.last_logits[r * vocab..(r + 1) * vocab];
    let scale = want.iter().fold(0f64, |a, &w| a.max(w.abs()));
    let diff = logits
        .iter()
        .zip(want)
        .fold(0f64, |a, (&g, &w)| a.max((f64::from(g) - w).abs()));
    diff / scale
}

/// Follows the oracle's greedy continuation from `logits` through `step`,
/// stopping at a mismatch the oracle's own margin explains; returns how
/// many tokens matched.
fn follow_greedy(
    golden: &Golden,
    margin: f64,
    logits: &mut [f32],
    mut step: impl FnMut(u32, &mut [f32]),
) -> usize {
    let mut next = argmax(logits);
    let mut produced = 0usize;
    for (i, &want) in golden.greedy.iter().enumerate() {
        if next != want {
            assert!(
                golden.greedy_margin[i] < margin,
                "greedy token {i}: {next} vs {want}"
            );
            eprintln!("greedy token {i}: within the oracle's margin, stopping");
            break;
        }
        produced += 1;
        step(next, logits);
        next = argmax(logits);
    }
    produced
}

/// The CPU decode token by token from the embedding, against the oracle:
/// the full logits of the recorded rows, every log-normalizer and target
/// log-probability, every top-1 and the greedy continuation; then the
/// prompt prefilled in one pass, whose last row is the decode's.
/// The original repository's config and `pytorch_model.bin` load the model
/// its `-hf` twin loads, bit for bit: the same embedding and every layer's
/// weights.
#[test]
#[ignore = "needs both official checkpoints (see the file header)"]
fn mamba_130m_original_bin_is_its_hf_twin() {
    let root = official_dir("MAMBA1_OFFICIAL_DIR", ".cache/mamba1-official");
    let a = load_hf(&root.join("mamba-130m")).unwrap();
    let b = load_hf(&root.join("mamba-130m-hf")).unwrap();
    assert_eq!(
        (a.vocab_size, a.vocab_size_padded, a.d_model),
        (b.vocab_size, b.vocab_size_padded, b.d_model)
    );
    assert!(a.lm_head.is_none() && b.lm_head.is_none());
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert!(bits(&a.embed) == bits(&b.embed), "embedding");
    let (wa, wb) = (a.backbone.weights(), b.backbone.weights());
    assert!(bits(&wa.norm_f_weight) == bits(&wb.norm_f_weight), "norm_f");
    assert_eq!(wa.layers.len(), wb.layers.len());
    for (i, (la, lb)) in wa.layers.iter().zip(&wb.layers).enumerate() {
        for (name, x, y) in [
            ("norm", &la.norm_weight, &lb.norm_weight),
            ("in_proj", &la.in_proj_w, &lb.in_proj_w),
            ("conv", &la.conv1d_weight, &lb.conv1d_weight),
            ("conv bias", &la.conv1d_bias, &lb.conv1d_bias),
            ("x_proj", &la.x_proj_w, &lb.x_proj_w),
            ("dt_proj", &la.dt_proj_w, &lb.dt_proj_w),
            ("dt bias", &la.dt_proj_b, &lb.dt_proj_b),
            ("A_log", &la.a_log, &lb.a_log),
            ("D", &la.d_param, &lb.d_param),
            ("out_proj", &la.out_proj_w, &lb.out_proj_w),
        ] {
            assert!(bits(x) == bits(y), "layer {i} {name}");
        }
    }
}

#[test]
#[ignore = "needs the official checkpoint and its goldens (see the file header)"]
fn mamba_130m_cpu_matches_the_f64_oracle() {
    let (model, golden) = checkpoint();
    let backbone = &model.backbone;
    let d = model.d_model;
    let vocab = model.vocab_size;
    let mut head = Head::new(&model);
    let mut state = backbone.alloc_state();
    let mut scratch = backbone.alloc_scratch();
    let mut hidden = vec![0f32; d];
    let mut logits = vec![0f32; vocab];

    let (mut worst_rel, mut worst_lse, mut worst_lp) = (0f64, 0f64, 0f64);
    for (t, &token) in golden.ids.iter().enumerate() {
        backbone.forward_step(embed(&model, token), &mut hidden, &mut state, &mut scratch);
        head.logits(&hidden, &mut logits);
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
            worst_rel = worst_rel.max(row_error(&golden, t, vocab, &logits));
        }
    }
    let last_hidden = hidden.clone();
    eprintln!(
        "mamba-130m CPU f32 vs f64: max|dlogit|/max|logit| = {worst_rel:.3e}, \
         max|dlse| = {worst_lse:.3e}, max|dlogprob| = {worst_lp:.3e}"
    );
    assert!(worst_rel < 1e-4, "logits rel {worst_rel:.3e}");
    assert!(worst_lp < 1e-3, "log-prob {worst_lp:.3e}");

    let produced = follow_greedy(&golden, 1e-3, &mut logits, |token, logits| {
        backbone.forward_step(embed(&model, token), &mut hidden, &mut state, &mut scratch);
        head.logits(&hidden, logits);
    });
    eprintln!("mamba-130m CPU greedy: {produced} tokens match the oracle");

    // The prompt in one pass: its last row is the decode chain's.
    let t_len = golden.ids.len();
    let prompt: Vec<f32> = golden
        .ids
        .iter()
        .flat_map(|&tok| embed(&model, tok).iter().copied())
        .collect();
    let mut st = backbone.alloc_state();
    let mut ps = backbone.alloc_prefill_scratch(t_len);
    let mut rows = vec![0f32; t_len * d];
    backbone.forward_prefill(
        &prompt,
        &mut rows,
        &mut st,
        &mut ps,
        t_len,
        PrefillMode::Single,
    );
    let last = &rows[(t_len - 1) * d..];
    let rel = last
        .iter()
        .zip(&last_hidden)
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    eprintln!("mamba-130m CPU prefill last row vs decode: max|d| = {rel:.3e}");
    assert!(rel < 1e-4, "prefill last row {rel:.3e}");
}

/// The GPU decode in `dtype`, token by token from the embedding, as logits
/// of every prompt position.
#[cfg(feature = "cuda")]
fn gpu_decode_rows(
    model: &HfModel,
    golden: &Golden,
    dtype: mamba_rs::gpu::dtype::WeightDtype,
) -> (
    mamba_rs::mamba1::gpu::inference::MambaGpuBackbone,
    Vec<Vec<f32>>,
) {
    use mamba_rs::mamba1::gpu::inference::MambaGpuBackbone;

    let d = model.d_model;
    let cfg = *model.backbone.config();
    let mut gpu = MambaGpuBackbone::new(
        model.backbone.weights(),
        cfg,
        d,
        1,
        GpuOptions::default().dtype(dtype),
    )
    .unwrap();
    let mut head = Head::new(model);
    let mut hidden = vec![0f32; d];
    let rows = golden
        .ids
        .iter()
        .map(|&token| {
            gpu.step(embed(model, token), &mut hidden).unwrap();
            let mut logits = vec![0f32; model.vocab_size];
            head.logits(&hidden, &mut logits);
            logits
        })
        .collect();
    (gpu, rows)
}

/// The f32 GPU engine against the oracle: the decode chain over the prompt
/// at every recorded row and the greedy continuation; then the prompt
/// prefilled in one pass, whose last row tracks the decode's.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the official checkpoint, its goldens and a CUDA device"]
fn mamba_130m_gpu_matches_the_f64_oracle() {
    use mamba_rs::gpu::buffers::GpuBuffer;
    use mamba_rs::gpu::dtype::WeightDtype;
    use mamba_rs::mamba1::gpu::inference::MambaGpuBackbone;

    let (model, golden) = checkpoint();
    let d = model.d_model;
    let vocab = model.vocab_size;
    let (mut gpu, rows) = gpu_decode_rows(&model, &golden, WeightDtype::F32);
    let mut worst = 0f64;
    for (t, logits) in rows.iter().enumerate() {
        assert_eq!(argmax(logits), golden.top_ids[t][0], "top-1 at {t}");
        if t >= golden.last_rows_from {
            worst = worst.max(row_error(&golden, t, vocab, logits));
        }
    }
    eprintln!("mamba-130m GPU f32 decode vs f64: max|dlogit|/max|logit| = {worst:.3e}");
    assert!(worst < 2e-3, "GPU logits rel {worst:.3e}");

    let mut head = Head::new(&model);
    let mut hidden = vec![0f32; d];
    let mut logits = rows.last().unwrap().clone();
    let produced = follow_greedy(&golden, 1e-2, &mut logits, |token, logits| {
        gpu.step(embed(&model, token), &mut hidden).unwrap();
        head.logits(&hidden, logits);
    });
    eprintln!("mamba-130m GPU f32 greedy: {produced} tokens match the oracle");

    let t_len = golden.ids.len();
    let cfg = *model.backbone.config();
    let mut fresh =
        MambaGpuBackbone::new(model.backbone.weights(), cfg, d, 1, GpuOptions::default()).unwrap();
    let prompt: Vec<f32> = golden
        .ids
        .iter()
        .flat_map(|&tok| embed(&model, tok).iter().copied())
        .collect();
    let mut input = GpuBuffer::zeros(fresh.stream(), prompt.len()).unwrap();
    input.upload(fresh.stream(), &prompt).unwrap();
    let mut scratch = fresh.alloc_prefill_scratch(t_len).unwrap();
    fresh.prefill_sequence(&input, &mut scratch).unwrap();
    let mut last = vec![0f32; d];
    fresh.download_temporal(&mut last).unwrap();
    let mut prefill_logits = vec![0f32; vocab];
    head.logits(&last, &mut prefill_logits);
    let rel = row_error(&golden, t_len - 1, vocab, &prefill_logits);
    eprintln!("mamba-130m GPU f32 prefill last row vs f64: max|dlogit|/max|logit| = {rel:.3e}");
    assert_eq!(argmax(&prefill_logits), golden.top_ids[t_len - 1][0]);
    assert!(rel < 2e-3, "GPU prefill logits rel {rel:.3e}");
}

/// The bf16 engine on the oracle's prompt: the mean prompt NLL within a
/// few hundredths of a nat and the greedy continuation where the oracle's
/// margin clears bf16 noise.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the official checkpoint, its goldens and a CUDA device"]
fn mamba_130m_gpu_bf16_tracks_the_f64_oracle() {
    use mamba_rs::gpu::dtype::WeightDtype;

    let (model, golden) = checkpoint();
    let d = model.d_model;
    let t_len = golden.ids.len();
    let (mut gpu, rows) = gpu_decode_rows(&model, &golden, WeightDtype::Bf16);
    let (mut nll, mut nll_oracle) = (0f64, 0f64);
    for t in 0..t_len - 1 {
        let logits = &rows[t];
        nll -= f64::from(logits[golden.ids[t + 1] as usize]) - logsumexp(logits);
        nll_oracle -= golden.target_logprob[t];
    }
    let n = (t_len - 1) as f64;
    let (nll, nll_oracle) = (nll / n, nll_oracle / n);
    eprintln!("mamba-130m GPU bf16 decode: mean NLL {nll:.4} vs oracle {nll_oracle:.4}");
    assert!(
        (nll - nll_oracle).abs() < 3e-2,
        "bf16 NLL {nll:.4} vs {nll_oracle:.4}"
    );
    let mut head = Head::new(&model);
    let mut hidden = vec![0f32; d];
    let mut logits = rows.last().unwrap().clone();
    let produced = follow_greedy(&golden, 0.25, &mut logits, |token, logits| {
        gpu.step(embed(&model, token), &mut hidden).unwrap();
        head.logits(&hidden, logits);
    });
    eprintln!("mamba-130m GPU bf16 greedy: {produced} tokens match the oracle");
}
