#![cfg(all(feature = "hf", feature = "cuda"))]

use std::time::Instant;

fn find_model_dir(name: &str) -> Option<std::path::PathBuf> {
    let cache = std::path::Path::new("/root/.cache/huggingface/hub");
    for entry in std::fs::read_dir(cache).ok()? {
        let entry = entry.ok()?;
        let fname = entry.file_name().into_string().ok()?;
        if fname.contains(name) {
            let snaps = entry.path().join("snapshots");
            if snaps.exists() {
                let snap = std::fs::read_dir(&snaps).ok()?.next()?.ok()?;
                return Some(snap.path());
            }
        }
    }
    None
}

#[test]
#[ignore = "requires downloaded mamba-130m-hf model"]
fn test_gpu_lm_130m() {
    let dir = find_model_dir("mamba-130m-hf").expect("model not in cache");

    let mut lm = mamba_rs::module::gpu_lm::GpuMambaLM::from_hf(&dir, 0).unwrap();

    let params = mamba_rs::module::sample::SampleParams {
        temperature: 0.0,
        max_tokens: 20,
        ..Default::default()
    };

    let tokens = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
    assert_eq!(tokens.len(), 20);
    for &t in &tokens {
        assert!((t as usize) < 50280);
    }

    // Determinism
    lm.reset().unwrap();
    let tokens2 = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
    assert_eq!(tokens, tokens2, "GPU greedy must be deterministic");

    let t = Instant::now();
    lm.reset().unwrap();
    let _ = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
    let tps = 20.0 / t.elapsed().as_secs_f64();
    eprintln!("GPU LM: {:.1} tok/s", tps);
}
