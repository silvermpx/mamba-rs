//! Diagnostic test for the bf16 batch-size divergence. Varies batch
//! size and slot position to narrow down whether the bug is:
//!   (a) shape-specific (triggers at batch >= threshold)
//!   (b) slot-position-specific (triggers at some slot)
//!   (c) input-specific (triggers on specific token IDs regardless of batch)

#![cfg(all(feature = "cuda", feature = "hf"))]

use std::path::PathBuf;

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::module::gpu_lm::GpuMambaLM;
use mamba_rs::module::sample::SampleParams;

fn find_model_dir(name: &str) -> Option<PathBuf> {
    for base in [
        "/root/.cache/huggingface/hub",
        "/home/silvermpx/.cache/huggingface/hub",
    ] {
        let cache = std::path::Path::new(base);
        if !cache.exists() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(cache) {
            for entry in entries.flatten() {
                if let Ok(fname) = entry.file_name().into_string()
                    && fname.contains(name)
                {
                    let snaps = entry.path().join("snapshots");
                    if snaps.exists()
                        && let Ok(mut snap_iter) = std::fs::read_dir(&snaps)
                        && let Some(Ok(snap)) = snap_iter.next()
                    {
                        return Some(snap.path());
                    }
                }
            }
        }
    }
    None
}

fn kl(p: &[f32], q: &[f32]) -> f32 {
    let pmax = p.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let qmax = q.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut ps = 0.0f64;
    let mut qs = 0.0f64;
    for (&pl, &ql) in p.iter().zip(q) {
        ps += ((pl as f64) - pmax).exp();
        qs += ((ql as f64) - qmax).exp();
    }
    let lps = ps.ln();
    let lqs = qs.ln();
    let mut k = 0.0f64;
    for (&pl, &ql) in p.iter().zip(q) {
        let lp = ((pl as f64) - pmax) - lps;
        let lq = ((ql as f64) - qmax) - lqs;
        let pv = lp.exp();
        if pv > 1e-30 {
            k += pv * (lp - lq);
        }
    }
    k as f32
}

#[test]
#[ignore]
fn bf16_batch_slot_matrix() {
    let dir = match find_model_dir("mamba-130m-hf") {
        Some(d) => d,
        None => {
            eprintln!("[skip] no HF cache");
            return;
        }
    };
    let params = SampleParams {
        temperature: 0.0,
        max_tokens: 1,
        ..Default::default()
    };

    // Reference: batch=1, [100..104].
    let mut lm1 =
        GpuMambaLM::from_hf_with_dtype_batch(&dir, 0, WeightDtype::Bf16, 1).unwrap();
    lm1.generate(&[100, 101, 102, 103, 104], &params).unwrap();
    let ref_logits = lm1.last_logits(0).to_vec();
    eprintln!("reference b=1 ok");

    // Matrix of (batch_size, slot_for_[100..104]).
    let cases: Vec<(usize, usize, Vec<&[u32]>)> = vec![
        (2, 0, vec![&[100, 101, 102, 103, 104], &[1, 2, 3, 4, 5]]),
        (2, 1, vec![&[1, 2, 3, 4, 5], &[100, 101, 102, 103, 104]]),
        (
            3,
            0,
            vec![
                &[100, 101, 102, 103, 104],
                &[1, 2, 3, 4, 5],
                &[10, 20, 30, 40, 50],
            ],
        ),
        (
            3,
            2,
            vec![
                &[1, 2, 3, 4, 5],
                &[10, 20, 30, 40, 50],
                &[100, 101, 102, 103, 104],
            ],
        ),
        (
            4,
            0,
            vec![
                &[100, 101, 102, 103, 104],
                &[1, 2, 3, 4, 5],
                &[10, 20, 30, 40, 50],
                &[1, 2, 3, 4, 5],
            ],
        ),
        (
            4,
            2,
            vec![
                &[1, 2, 3, 4, 5],
                &[10, 20, 30, 40, 50],
                &[100, 101, 102, 103, 104],
                &[1, 2, 3, 4, 5],
            ],
        ),
    ];

    for (b, slot, prompts) in cases {
        let mut lm =
            GpuMambaLM::from_hf_with_dtype_batch(&dir, 0, WeightDtype::Bf16, b).unwrap();
        let ps = vec![params.clone(); b];
        lm.generate_batch(&prompts, &ps).unwrap();
        let slot_logits = lm.last_logits(slot).to_vec();
        let d = kl(&ref_logits, &slot_logits);
        eprintln!("b={b} slot={slot} [100..104]  KL vs b=1 = {d:.4e}");
    }
}
