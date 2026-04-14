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

fn bench_model(name: &str) {
    let dir = match find_model_dir(name) {
        Some(d) => d,
        None => {
            eprintln!("{}: NOT IN CACHE, skipping", name);
            return;
        }
    };

    let t = Instant::now();
    let mut lm = mamba_rs::module::gpu_lm::GpuMambaLM::from_hf(&dir, 0).unwrap();
    let load_ms = t.elapsed().as_millis();

    lm.capture_graph().unwrap();

    let params = mamba_rs::module::sample::SampleParams {
        temperature: 0.0,
        max_tokens: 100,
        ..Default::default()
    };

    // Warmup
    let _ = lm.generate(&[1, 2, 3], &params);
    lm.reset().unwrap();

    let t = Instant::now();
    let tokens = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
    let tps = tokens.len() as f64 / t.elapsed().as_secs_f64();
    eprintln!(
        "{}: d_model={} | load {}ms | {:.0} tok/s (graph)",
        name, lm.d_model, load_ms, tps
    );
}

#[test]
#[ignore]
fn bench_all_gpu() {
    bench_model("mamba-130m-hf");
    bench_model("mamba-370m-hf");
    bench_model("mamba-1.4b-hf");
    bench_model("mamba-2.8b-hf");
}
