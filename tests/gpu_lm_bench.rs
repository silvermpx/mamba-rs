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
fn bench_gpu_lm_130m_no_graph() {
    let dir = find_model_dir("mamba-130m-hf").unwrap();
    let mut lm = mamba_rs::module::gpu_lm::GpuMambaLM::from_hf(&dir, 0).unwrap();
    let params = mamba_rs::module::sample::SampleParams {
        temperature: 0.0,
        max_tokens: 100,
        ..Default::default()
    };
    // warmup
    let _ = lm.generate(&[1, 2, 3], &params);
    lm.reset().unwrap();

    let t = Instant::now();
    let tokens = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
    let tps = tokens.len() as f64 / t.elapsed().as_secs_f64();
    eprintln!("NO GRAPH: {} tokens, {:.1} tok/s", tokens.len(), tps);
}
