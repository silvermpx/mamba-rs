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
fn profile_gpu_backbone_only() {
    // Test backbone-only speed (no lm_head) to isolate bottleneck
    use mamba_rs::MambaConfig;
    use mamba_rs::MambaWeights;
    use mamba_rs::gpu::inference::GpuMambaBackbone;

    // Use 130m-hf config
    let cfg = MambaConfig {
        d_model: 768,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        n_layers: 24,
        ..Default::default()
    };
    let weights = MambaWeights::init(&cfg, 768, 42);
    let mut bb = GpuMambaBackbone::new(0, &weights, cfg, 768, 1).unwrap();

    let input = vec![0.1f32; 768];
    let mut output = vec![0.0f32; 768];

    // Warmup
    for _ in 0..10 {
        bb.step(&input, &mut output).unwrap();
    }
    bb.reset().unwrap();

    // Without CUDA Graph
    let n = 200;
    let t = Instant::now();
    for _ in 0..n {
        bb.step(&input, &mut output).unwrap();
    }
    let no_graph = t.elapsed().as_secs_f64() / n as f64;
    eprintln!(
        "Backbone only (no graph): {:.1}us/step = {:.0} tok/s",
        no_graph * 1e6,
        1.0 / no_graph
    );

    // With CUDA Graph
    bb.reset().unwrap();
    bb.capture_graph().unwrap();

    let t = Instant::now();
    for _ in 0..n {
        bb.step(&input, &mut output).unwrap();
    }
    let with_graph = t.elapsed().as_secs_f64() / n as f64;
    eprintln!(
        "Backbone only (with graph): {:.1}us/step = {:.0} tok/s",
        with_graph * 1e6,
        1.0 / with_graph
    );

    eprintln!("Graph speedup: {:.1}x", no_graph / with_graph);
}
