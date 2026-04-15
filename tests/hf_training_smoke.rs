//! Gap 2: real HF checkpoint training smoke.
//!
//! Current coverage wires the HF loader into inference only
//! (`gpu_lm_test.rs`, `gpu_bf16_parity.rs`). This test drives the full
//! `MambaTrainer` through a few gradient steps starting from real
//! `state-spaces/mamba-130m-hf` weights and asserts that:
//!
//! 1. The loader → trainer hand-off succeeds with the real HF shapes
//!    (d_model=768, n_layers=24, d_state=16, d_conv=4, expand=2) — i.e.
//!    `MambaWeights::layers[*].a_neg` is populated when `a_log` is
//!    loaded from `backbone.layers.N.mixer.A_log`.
//! 2. Forward + backward + AdamW run end-to-end without NaN/Inf on real
//!    weights (synthetic random weights can hide issues that only appear
//!    with the real checkpoint's activation statistics).
//! 3. A CUDA Graph capture over the real-weight trainer replays
//!    correctly — the graph captured-pointer invariants hold at
//!    130m scale.
//!
//! `#[ignore]` by default because it needs ~500 MB of HF cache and a
//! GPU with ≥ 2 GB VRAM. Run with:
//!
//!   cargo test --release --features "cuda hf" --test hf_training_smoke \
//!       -- --ignored --nocapture

#![cfg(all(feature = "cuda", feature = "hf"))]

use std::path::PathBuf;

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

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
        })
        .collect()
}

#[test]
#[ignore]
fn hf_training_130m_bf16_smoke() {
    use mamba_rs::hf::load::load_hf;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::mamba_ssm::gpu::trainer::MambaTrainer;

    let dir = match find_model_dir("mamba-130m-hf") {
        Some(d) => d,
        None => {
            eprintln!("[skip] mamba-130m-hf not in HF cache");
            return;
        }
    };
    let hf = load_hf(&dir).expect("load mamba-130m-hf");
    let cfg = *hf.backbone.config();
    let input_dim = cfg.d_model;
    eprintln!(
        "loaded mamba-130m-hf: d_model={} n_layers={} d_state={} d_conv={} expand={}",
        cfg.d_model, cfg.n_layers, cfg.d_state, cfg.d_conv, cfg.expand
    );

    // Sanity-check the loader populated `a_neg` for every layer.
    for (li, lw) in hf.backbone.weights().layers.iter().enumerate() {
        assert!(
            lw.a_neg.iter().all(|&v| v.is_finite() && v <= 0.0),
            "L{li}.a_neg malformed — HF a_log → a_neg conversion broken"
        );
    }

    let batch = 1;
    let seq_len = 8;
    let n = batch * seq_len * input_dim;

    // lr=1e-6 with bf16 gives small but finite updates on real-weight grads.
    let mut trainer = MambaTrainer::new_full(
        0,
        hf.backbone.weights(),
        cfg,
        input_dim,
        batch,
        seq_len,
        WeightDtype::Bf16,
        1e-6,
        0.0,
    )
    .expect("construct 130m bf16 trainer");

    let before = trainer.snapshot_master().expect("snapshot pre");

    // Eager warmup + steps. input_scale=0.05 is roughly the magnitude of a
    // token embedding after layer norm in Mamba-130m; keeps the bf16 forward
    // well inside range without needing the tokenizer.
    let input_scale = 0.05f32;
    let grad_scale = 0.01f32;
    for s in 0..3 {
        let m = trainer
            .step(
                &det(n, 0xA0 + s, input_scale),
                &det(n, 0xB0 + s, grad_scale),
            )
            .expect("eager step on 130m weights");
        assert!(!m.graph_replayed);
        eprintln!("  eager step {s} ok (step={})", m.step);
    }

    // Capture + replay — full 24-layer graph round-trip.
    trainer
        .capture_graph()
        .expect("capture 24-layer graph on real weights");
    assert!(trainer.has_graph());
    for s in 0..3 {
        let m = trainer
            .step(
                &det(n, 0xC0 + s, input_scale),
                &det(n, 0xD0 + s, grad_scale),
            )
            .expect("graph step on 130m weights");
        assert!(m.graph_replayed);
    }

    // Post-condition: weights finite and moved on at least one layer.
    let after = trainer.snapshot_master().expect("snapshot post");
    let mut any_change = false;
    for (li, lw) in after.layers.iter().enumerate() {
        for (name, slice) in [
            ("in_proj_w", lw.in_proj_w.as_slice()),
            ("out_proj_w", lw.out_proj_w.as_slice()),
            ("x_proj_w", lw.x_proj_w.as_slice()),
            ("dt_proj_w", lw.dt_proj_w.as_slice()),
            ("a_log", lw.a_log.as_slice()),
        ] {
            assert!(
                slice.iter().all(|v| v.is_finite()),
                "L{li}.{name} non-finite after 6 training steps on real weights"
            );
        }
        let diff = before.layers[li]
            .in_proj_w
            .iter()
            .zip(&lw.in_proj_w)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if diff > 0.0 {
            any_change = true;
        }
    }
    assert!(
        any_change,
        "no layer's in_proj_w moved — training path is a no-op on real weights"
    );
}
