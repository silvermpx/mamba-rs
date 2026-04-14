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
#[ignore]
fn test_gpu_lm_bf16_130m_loads_and_generates() {
    let dir = find_model_dir("mamba-130m-hf").expect("model not in cache");
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::module::gpu_lm::GpuMambaLM;
    use mamba_rs::module::sample::SampleParams;

    let t_load = Instant::now();
    let mut lm = GpuMambaLM::from_hf_with_dtype(&dir, 0, WeightDtype::Bf16).unwrap();
    eprintln!(
        "GpuMambaLM bf16 loaded in {}ms (vocab={}, d_model={})",
        t_load.elapsed().as_millis(),
        lm.vocab_size,
        lm.d_model
    );

    let params = SampleParams {
        temperature: 0.0,
        max_tokens: 20,
        ..Default::default()
    };

    let tokens = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
    assert_eq!(tokens.len(), 20);
    for &t in &tokens {
        assert!((t as usize) < lm.vocab_size, "token {t} out of range");
    }

    // Determinism
    lm.reset().unwrap();
    let tokens2 = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
    assert_eq!(tokens, tokens2, "bf16 greedy must be deterministic");

    // Benchmark
    let t = Instant::now();
    lm.reset().unwrap();
    let _ = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
    let tps = 20.0 / t.elapsed().as_secs_f64();
    eprintln!("GPU LM bf16: {:.1} tok/s", tps);
}
