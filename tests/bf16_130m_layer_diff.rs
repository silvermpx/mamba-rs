//! bf16 vs f32 divergence harness for mamba-130m-hf. Uses only public APIs.
//!
//! Runs prompts of increasing length; after each, calls compute_logits (via
//! generate with max_tokens=0) and compares last_logits between f32 and bf16.

#![cfg(all(feature = "hf", feature = "cuda"))]

use std::path::PathBuf;

fn find_model_dir(name: &str) -> Option<PathBuf> {
    let cache = std::path::Path::new("/root/.cache/huggingface/hub");
    for entry in std::fs::read_dir(cache).ok()?.flatten() {
        if let Ok(fname) = entry.file_name().into_string()
            && fname.contains(name)
        {
            let snaps = entry.path().join("snapshots");
            if snaps.exists()
                && let Ok(mut it) = std::fs::read_dir(&snaps)
                && let Some(Ok(snap)) = it.next()
            {
                return Some(snap.path());
            }
        }
    }
    None
}

fn stats(buf: &[f32]) -> (f32, usize, usize) {
    let mut max_abs = 0f32;
    let mut nan = 0usize;
    let mut inf = 0usize;
    for &v in buf {
        if v.is_nan() {
            nan += 1;
        } else if v.is_infinite() {
            inf += 1;
        } else {
            max_abs = max_abs.max(v.abs());
        }
    }
    (max_abs, nan, inf)
}

fn top_k_idx(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(k);
    pairs
}

#[test]
#[ignore]
fn bf16_130m_divergence_bisect() {
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::module::gpu_lm::GpuMambaLM;
    use mamba_rs::module::sample::SampleParams;

    let dir = match find_model_dir("mamba-130m-hf") {
        Some(d) => d,
        None => {
            eprintln!("[skip] mamba-130m-hf not in HF cache");
            return;
        }
    };

    let mut lm_f32 = GpuMambaLM::from_hf_with_dtype(&dir, 0, WeightDtype::F32).unwrap();
    let mut lm_bf16 = GpuMambaLM::from_hf_with_dtype(&dir, 0, WeightDtype::Bf16).unwrap();

    // Drive with generate(max_tokens=1) on suffixes of the prompt so each
    // run builds state from scratch (reset() inside generate_streaming),
    // process len tokens of prefill, and leaves logits in last_logits().
    let full_prompt: &[u32] = &[1, 2, 3, 4, 5, 6];

    eprintln!(
        "\n=== 130m bf16 vs f32 logits divergence (prompt length → logits at last position) ==="
    );
    eprintln!(
        "{:>4} | {:>11} {:>11} | {:>5} {:>5} | {:>11} {:>10} | {}",
        "len",
        "f32 |L|∞",
        "bf16 |L|∞",
        "f32_nan",
        "bf_nan",
        "L∞ diff",
        "top1 same",
        "top3 f32 → bf16"
    );

    let d = lm_f32.d_model;
    let mut t_f32 = vec![0f32; d];
    let mut t_bf = vec![0f32; d];

    for len in 1..=full_prompt.len() {
        let prompt = &full_prompt[..len];
        let params = SampleParams {
            temperature: 0.0,
            max_tokens: 0,
            ..Default::default()
        };
        let _ = lm_f32.generate(prompt, &params).unwrap();
        let _ = lm_bf16.generate(prompt, &params).unwrap();

        // Temporal diff (backbone output, before lm_head).
        lm_f32.debug_download_temporal(&mut t_f32).unwrap();
        lm_bf16.debug_download_temporal(&mut t_bf).unwrap();
        let t_diff = t_f32
            .iter()
            .zip(&t_bf)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let (tf_max, tf_nan, _) = stats(&t_f32);
        let (tb_max, tb_nan, _) = stats(&t_bf);

        let f32_logits = lm_f32.last_logits(0).to_vec();
        let bf_logits = lm_bf16.last_logits(0).to_vec();
        let ldiff = f32_logits
            .iter()
            .zip(&bf_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let f_top = top_k_idx(&f32_logits, 3);
        let b_top = top_k_idx(&bf_logits, 3);
        eprintln!(
            "{:>4} | temp f32|x|∞={:>9.3e} bf16|x|∞={:>9.3e} diff={:>9.3e} nan={}+{} | logit Δ={:>9.3e} top1same={} | {:?} → {:?}",
            len,
            tf_max,
            tb_max,
            t_diff,
            tf_nan,
            tb_nan,
            ldiff,
            f_top.first().map(|p| p.0) == b_top.first().map(|p| p.0),
            f_top.iter().map(|p| p.0).collect::<Vec<_>>(),
            b_top.iter().map(|p| p.0).collect::<Vec<_>>(),
        );
    }
}
