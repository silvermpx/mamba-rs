//! bf16 vs f32 benchmarks for Mamba-1 (mamba-130m-hf) and Mamba-3 (synthetic).
//!
//! Ignored by default — run on a machine with a capable NVIDIA GPU via:
//!   cargo test --features cuda,hf,cli --release --test bench_bf16_vs_f32 -- --ignored --nocapture

#![cfg(all(feature = "hf", feature = "cuda"))]

use std::time::Instant;

fn find_model_dir(name: &str) -> Option<std::path::PathBuf> {
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

// ═══════════════════ Mamba-1 benchmarks ═══════════════════

fn bench_m1_dtype(
    dir: &std::path::Path,
    dtype: mamba_rs::mamba_ssm::gpu::dtype::WeightDtype,
    label: &str,
    use_graph: bool,
) {
    use mamba_rs::module::gpu_lm::GpuMambaLM;
    use mamba_rs::module::sample::SampleParams;

    let t_load = Instant::now();
    let mut lm = GpuMambaLM::from_hf_with_dtype(dir, 0, dtype).unwrap();
    let load_ms = t_load.elapsed().as_millis();

    if use_graph {
        lm.capture_graph().unwrap();
    }

    let params = SampleParams {
        temperature: 0.0,
        max_tokens: 100,
        ..Default::default()
    };

    // Warmup
    let _ = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
    lm.reset().unwrap();

    // Measure
    let t = Instant::now();
    let tokens = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
    let secs = t.elapsed().as_secs_f64();
    let tps = tokens.len() as f64 / secs;
    eprintln!(
        "  M1 130m {label}{}: load {load_ms}ms | {} tok | {:.3}s | {:.1} tok/s | {:.2} ms/tok",
        if use_graph { " +graph" } else { "" },
        tokens.len(),
        secs,
        tps,
        1000.0 * secs / tokens.len() as f64
    );
}

#[test]
#[ignore]
fn bench_m1_bf16_vs_f32_130m() {
    let dir = match find_model_dir("mamba-130m-hf") {
        Some(d) => d,
        None => {
            eprintln!("[skip] mamba-130m-hf not in HF cache");
            return;
        }
    };
    eprintln!("\n=== Mamba-1 130m bf16 vs f32 (RTX 6000 Ada) ===");
    for use_graph in [false, true] {
        use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
        bench_m1_dtype(&dir, WeightDtype::F32, "f32 ", use_graph);
        bench_m1_dtype(&dir, WeightDtype::Bf16, "bf16", use_graph);
        bench_m1_dtype(&dir, WeightDtype::F16, "f16 ", use_graph);
    }
}

#[test]
#[ignore]
fn bench_m1_weight_compression_ratio() {
    let dir = match find_model_dir("mamba-130m-hf") {
        Some(d) => d,
        None => {
            eprintln!("[skip] mamba-130m-hf not in HF cache");
            return;
        }
    };
    use mamba_rs::hf::load::load_hf;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    let hf = load_hf(&dir).unwrap();
    // Count total weight elements in the backbone + embed + lm_head.
    let mut total_elems: usize = hf.embed.len();
    if let Some(ref lm) = hf.lm_head {
        total_elems += lm.len();
    }
    let w = hf.backbone.weights();
    total_elems += w.input_proj_w.len() + w.input_proj_b.len() + w.norm_f_weight.len();
    for lw in &w.layers {
        total_elems += lw.norm_weight.len()
            + lw.in_proj_w.len()
            + lw.conv1d_weight.len()
            + lw.conv1d_bias.len()
            + lw.x_proj_w.len()
            + lw.dt_proj_w.len()
            + lw.dt_proj_b.len()
            + lw.a_log.len()
            + lw.d_param.len()
            + lw.out_proj_w.len();
    }

    eprintln!("\n=== Mamba-1 130m weight storage (theoretical) ===");
    for (dtype, label) in [
        (WeightDtype::F32, "f32 "),
        (WeightDtype::Bf16, "bf16"),
        (WeightDtype::F16, "f16 "),
    ] {
        let bytes = total_elems * dtype.size_bytes();
        let mb = bytes as f64 / (1024.0 * 1024.0);
        let ratio = bytes as f64 / (total_elems * 4) as f64;
        eprintln!("  {label}: {:>6.1} MB ({:.2}x vs f32)", mb, ratio,);
    }
}

// ═══════════════════ Mamba-3 benchmarks (synthetic weights) ═══════════════════

fn m3_bench_cfg() -> mamba_rs::mamba3_siso::config::Mamba3Config {
    mamba_rs::mamba3_siso::config::Mamba3Config {
        d_model: 256,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 8,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
    }
}

fn bench_m3_dtype(
    dtype: mamba_rs::mamba_ssm::gpu::dtype::WeightDtype,
    label: &str,
    use_graph: bool,
) {
    use mamba_rs::mamba3_siso::weights::Mamba3Weights;
    use mamba_rs::module::gpu_lm3::GpuMamba3LM;
    use mamba_rs::module::sample::SampleParams;

    let cfg = m3_bench_cfg();
    let vocab_size = 256;
    let d = cfg.d_model;

    let t_build = Instant::now();
    let weights = Mamba3Weights::init(&cfg, d, 0xB00B);
    let mut embed = vec![0.0f32; vocab_size * d];
    let mut seed: u64 = 0xBEEF;
    for v in embed.iter_mut() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        *v = (((seed & 0xFFFFFF) as f32 / 16777216.0) - 0.5) * 0.1;
    }
    let mut lm =
        GpuMamba3LM::from_weights_with_dtype(&weights, cfg, embed, None, vocab_size, 0, dtype)
            .unwrap();
    let build_ms = t_build.elapsed().as_millis();

    if use_graph {
        lm.capture_graph().unwrap();
    }

    let params = SampleParams {
        temperature: 0.0,
        max_tokens: 100,
        ..Default::default()
    };
    let _ = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
    lm.reset().unwrap();

    let t = Instant::now();
    let tokens = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
    let secs = t.elapsed().as_secs_f64();
    let tps = tokens.len() as f64 / secs;
    eprintln!(
        "  M3 syn  {label}{}: build {build_ms}ms | {} tok | {:.3}s | {:.1} tok/s | {:.2} ms/tok",
        if use_graph { " +graph" } else { "" },
        tokens.len(),
        secs,
        tps,
        1000.0 * secs / tokens.len() as f64
    );
}

#[test]
#[ignore]
fn bench_m3_bf16_vs_f32_synthetic() {
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    eprintln!("\n=== Mamba-3 synthetic (d_model=256, n_layers=8, nh=32) bf16 vs f32 ===");
    for use_graph in [false, true] {
        bench_m3_dtype(WeightDtype::F32, "f32 ", use_graph);
        bench_m3_dtype(WeightDtype::Bf16, "bf16", use_graph);
        bench_m3_dtype(WeightDtype::F16, "f16 ", use_graph);
    }
}
