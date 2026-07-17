//! Classifier-shape VRAM footprint probe (manual: --ignored).
//!
//! Constructs the bf16 trainer at the doc-101 classifier center shape
//! (d_model=384 x 24 layers, d_state=16, T=4617, input_dim=1024 = 32x32
//! grayscale patches, trainable input_proj) across micro-batch sizes and
//! reports nvidia-smi memory after construction and after one step.
//! Settles the "B=4 fits 32 GB?" question empirically.

#![cfg(feature = "cuda")]

use mamba_rs::config::MambaConfig;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
use mamba_rs::weights::MambaWeights;

fn vram_used_mb() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "?".into())
}

#[test]
#[ignore]
fn classifier_shape_memory_footprint() {
    let cfg = MambaConfig {
        d_model: 384,
        n_layers: 24,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    };
    let input_dim = 1024usize; // 32x32 grayscale patch
    let seq_len = 4617usize;
    let w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);

    for batch in [1usize, 2, 4] {
        eprintln!("=== batch {batch}: baseline {} MiB", vram_used_mb());
        let session = TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr: 5e-4,
            weight_decay: 0.05,
        };
        match MambaTrainer::new_full(0, &w, cfg, session, WeightDtype::Bf16) {
            Ok(mut t) => {
                eprintln!("    constructed: {} MiB", vram_used_mb());
                let input = vec![0.01f32; batch * seq_len * input_dim];
                let d_temporal = vec![0.001f32; batch * seq_len * cfg.d_model];
                match t.step(&input, &d_temporal) {
                    Ok(_) => eprintln!("    after step: {} MiB — B{batch} FITS", vram_used_mb()),
                    Err(e) => eprintln!("    step FAILED at B{batch}: {e}"),
                }
            }
            Err(e) => eprintln!("    construction FAILED at B{batch}: {e}"),
        }
        // Trainer dropped here — next iteration starts clean.
    }
}
