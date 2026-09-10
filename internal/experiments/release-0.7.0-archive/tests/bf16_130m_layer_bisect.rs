//! Per-layer bf16 vs f32 bisect on mamba-130m-hf. For each layer N=1..n_layers,
//! reset both LMs, run `debug_step_one_token` with `layer_limit=N`, download
//! the post-layer-N residual, and report the L∞ / max-rel diff. The first N
//! where the diff blows up identifies the culprit layer.

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

fn diff(a: &[f32], b: &[f32]) -> (f32, f32, f32, f32) {
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut sum_sq = 0f64;
    let mut max_a = 0f32;
    let mut max_b = 0f32;
    for (&x, &y) in a.iter().zip(b) {
        if x.is_finite() {
            max_a = max_a.max(x.abs());
        }
        if y.is_finite() {
            max_b = max_b.max(y.abs());
        }
        if x.is_finite() && y.is_finite() {
            let d = (x - y).abs();
            max_abs = max_abs.max(d);
            let denom = x.abs().max(y.abs()).max(1e-6);
            max_rel = max_rel.max(d / denom);
            sum_sq += (d as f64) * (d as f64);
        }
    }
    let _ = sum_sq;
    let _ = max_a;
    let _ = max_b;
    (max_abs, max_rel, max_a, max_b)
}

#[test]
#[ignore]
fn bf16_130m_per_layer_bisect() {
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::module::gpu_lm::GpuMambaLM;

    let dir = match find_model_dir("mamba-130m-hf") {
        Some(d) => d,
        None => {
            eprintln!("[skip] mamba-130m-hf not in HF cache");
            return;
        }
    };

    let mut lm_f32 = GpuMambaLM::from_hf_with_dtype(&dir, 0, WeightDtype::F32).unwrap();
    let mut lm_bf16 = GpuMambaLM::from_hf_with_dtype(&dir, 0, WeightDtype::Bf16).unwrap();

    let d = lm_f32.d_model;
    let mut out_f32 = vec![0f32; d];
    let mut out_bf = vec![0f32; d];

    eprintln!(
        "\n=== 130m bf16 vs f32 per-layer bisect (input = embed(1), 24 layers, d_model={d}) ==="
    );
    eprintln!(
        "{:>5} | {:>11} {:>11} | {:>11} {:>11}",
        "layer", "f32 |x|∞", "bf16 |x|∞", "L∞ diff", "max rel"
    );

    // n_layers for 130m = 24. We try 0..=24 (0 = before any layer, just embed
    // copy if applicable; 24 = full backbone except final norm_f).
    for n in 0..=24 {
        lm_f32.debug_step_one_token(1, n, &mut out_f32).unwrap();
        lm_bf16.debug_step_one_token(1, n, &mut out_bf).unwrap();
        let (l_inf, max_rel, ma, mb) = diff(&out_f32, &out_bf);
        eprintln!(
            "{:>5} | {:>11.4e} {:>11.4e} | {:>11.4e} {:>11.3}",
            n, ma, mb, l_inf, max_rel
        );
    }
}
