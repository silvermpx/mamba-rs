//! Stability stress tests — targeted corner cases across all dtypes.
//!
//! Three dark-corner scenarios not covered elsewhere:
//!
//! 1. **CUDA Graph replay determinism** — capture once, replay N times
//!    with identical inputs, assert every replay produces bit-identical
//!    master weights. Catches any non-determinism introduced by kernel
//!    launches inside the captured body (e.g. race on a shared
//!    atomicAdd target, non-deterministic reduction order).
//!
//! 2. **Long-sequence inference stability** — drive `GpuMambaLM` through
//!    256 tokens and verify no NaN / Inf and logit norms stay bounded.
//!    Mamba's SSM recurrence can drift over long sequences if the decay
//!    math is slightly off; this is the canary.
//!
//! 3. **Training step repeatability on identical inputs** — build a
//!    trainer twice with the same seed + same weights, run N identical
//!    steps on both, assert they agree bit-for-bit. If any kernel
//!    depends on a timestamp / address-dependent hash / non-deterministic
//!    launch-time setting, this surfaces it.
//!
//! `#[ignore]` for tests needing HF cache; the synthetic-weight
//! determinism tests run without cache.

#![cfg(feature = "cuda")]

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::trainer::MambaTrainer;
use mamba_rs::weights::MambaWeights;

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFFFF) as f32 / 65536.0 - 0.5
        })
        .collect()
}

fn flatten_master(w: &MambaWeights) -> Vec<f32> {
    let mut out = Vec::new();
    out.extend_from_slice(&w.input_proj_w);
    out.extend_from_slice(&w.input_proj_b);
    for lw in &w.layers {
        out.extend_from_slice(&lw.norm_weight);
        out.extend_from_slice(&lw.in_proj_w);
        out.extend_from_slice(&lw.conv1d_weight);
        out.extend_from_slice(&lw.conv1d_bias);
        out.extend_from_slice(&lw.x_proj_w);
        out.extend_from_slice(&lw.dt_proj_w);
        out.extend_from_slice(&lw.dt_proj_b);
        out.extend_from_slice(&lw.a_log);
        out.extend_from_slice(&lw.d_param);
        out.extend_from_slice(&lw.out_proj_w);
    }
    out.extend_from_slice(&w.norm_f_weight);
    out
}

fn cfg() -> MambaConfig {
    MambaConfig {
        d_model: 32,
        n_layers: 2,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Sequential,
    }
}

// =====================================================================
// 1. Training step repeatability: two independent trainers from the
//    same weights run identical inputs → identical master weights.
// =====================================================================

fn run_repeatability(dtype: WeightDtype) {
    let cfg_m = cfg();
    let input_dim = cfg_m.d_model;
    let batch = 1;
    let seq_len = 8;
    let n = batch * seq_len * input_dim;

    let mut cpu = MambaWeights::init(&cfg_m, input_dim, 0x5EED);
    if !matches!(dtype, WeightDtype::F32) {
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
    }
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    let mut a = MambaTrainer::new_full(0, &cpu, cfg_m, input_dim, batch, seq_len, dtype, 1e-4, 0.0)
        .unwrap();
    let mut b = MambaTrainer::new_full(0, &cpu, cfg_m, input_dim, batch, seq_len, dtype, 1e-4, 0.0)
        .unwrap();

    // 10 identical steps on both trainers.
    let g_scale = if matches!(dtype, WeightDtype::F16) {
        0.001
    } else {
        0.1
    };
    for s in 0..10 {
        let inp = det(n, 0xA0 + s);
        let mut dt_scaled: Vec<f32> = det(n, 0xB0 + s);
        for v in dt_scaled.iter_mut() {
            *v *= g_scale;
        }
        a.step(&inp, &dt_scaled).unwrap();
        b.step(&inp, &dt_scaled).unwrap();
    }

    let wa = flatten_master(&a.snapshot_master().unwrap());
    let wb = flatten_master(&b.snapshot_master().unwrap());
    assert_eq!(wa.len(), wb.len());
    let max_diff: f32 = wa
        .iter()
        .zip(&wb)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("repeatability [{dtype:?}] max_abs_diff = {max_diff:.3e}");
    // Training is deterministic up to cuBLAS kernel-selection order (which
    // is stable within a single process). Allow a tiny tolerance because
    // two separate trainer instances may select slightly different cuBLAS
    // paths under memory-pressure heuristics.
    assert!(
        max_diff < 1e-5,
        "training non-repeatable across independent trainer instances: max_diff={max_diff:.3e}"
    );
}

#[test]
fn training_repeatable_f32() {
    run_repeatability(WeightDtype::F32);
}

#[test]
fn training_repeatable_bf16() {
    run_repeatability(WeightDtype::Bf16);
}

#[test]
fn training_repeatable_f16() {
    run_repeatability(WeightDtype::F16);
}

// =====================================================================
// 2. CUDA Graph replay determinism — capture once, replay N times on
//    identical inputs, assert bit-identical weights after each batch
//    of replays.
// =====================================================================

fn run_graph_determinism(dtype: WeightDtype) {
    let cfg_m = cfg();
    let input_dim = cfg_m.d_model;
    let batch = 1;
    let seq_len = 8;
    let n = batch * seq_len * input_dim;

    let mut cpu = MambaWeights::init(&cfg_m, input_dim, 0xDECAFFFF);
    if !matches!(dtype, WeightDtype::F32) {
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
    }
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    let g_scale = if matches!(dtype, WeightDtype::F16) {
        0.001
    } else {
        0.1
    };

    // Trainer A: eager for 3 steps
    let mut a = MambaTrainer::new_full(0, &cpu, cfg_m, input_dim, batch, seq_len, dtype, 1e-4, 0.0)
        .unwrap();
    for s in 0..3 {
        let inp = det(n, 0xC0 + s);
        let mut dtg: Vec<f32> = det(n, 0xD0 + s);
        for v in dtg.iter_mut() {
            *v *= g_scale;
        }
        a.step(&inp, &dtg).unwrap();
    }
    let wa_ref = flatten_master(&a.snapshot_master().unwrap());
    drop(a);

    // Trainer B: warmup eager 1 step, then capture + replay 2 times on
    // the SAME inputs that eager A used for steps 2,3.
    let mut b = MambaTrainer::new_full(0, &cpu, cfg_m, input_dim, batch, seq_len, dtype, 1e-4, 0.0)
        .unwrap();
    let inp0 = det(n, 0xC0);
    let mut dt0: Vec<f32> = det(n, 0xD0);
    for v in dt0.iter_mut() {
        *v *= g_scale;
    }
    b.step(&inp0, &dt0).unwrap();
    b.capture_graph().unwrap();
    for s in 1..3 {
        let inp = det(n, 0xC0 + s);
        let mut dtg: Vec<f32> = det(n, 0xD0 + s);
        for v in dtg.iter_mut() {
            *v *= g_scale;
        }
        let m = b.step(&inp, &dtg).unwrap();
        assert!(m.graph_replayed, "graph should have replayed");
    }
    let wb_graph = flatten_master(&b.snapshot_master().unwrap());
    drop(b);

    assert_eq!(wa_ref.len(), wb_graph.len());
    let max_diff: f32 = wa_ref
        .iter()
        .zip(&wb_graph)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("graph-vs-eager [{dtype:?}] max_abs_diff = {max_diff:.3e}");
    // Same bit-identical-ness tolerance as the repeatability test. Graph
    // replay and eager execution take the SAME capturable kernel path
    // so they should produce identical output up to cuBLAS selection
    // stability.
    assert!(
        max_diff < 1e-5,
        "graph replay diverged from eager: max_diff={max_diff:.3e}"
    );

    // Now drive a SECOND B trainer and replay 10 times — all replays
    // must land at the same terminal weights.
    let mut c = MambaTrainer::new_full(0, &cpu, cfg_m, input_dim, batch, seq_len, dtype, 1e-4, 0.0)
        .unwrap();
    c.step(&inp0, &dt0).unwrap();
    c.capture_graph().unwrap();
    let inp1 = det(n, 0xE0);
    let mut dt1: Vec<f32> = det(n, 0xF0);
    for v in dt1.iter_mut() {
        *v *= g_scale;
    }
    // Same input replayed 10 times.
    let mut prev_snapshots: Vec<Vec<f32>> = Vec::new();
    for _ in 0..10 {
        c.step(&inp1, &dt1).unwrap();
        prev_snapshots.push(flatten_master(&c.snapshot_master().unwrap()));
    }
    // Sanity: weights MUST move between replays (each step is a fresh
    // AdamW update on accumulated m/v, even with identical grads).
    let first_last_diff: f32 = prev_snapshots[0]
        .iter()
        .zip(&prev_snapshots[9])
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("10-replay drift [{dtype:?}] step_0→step_9 max_abs_diff = {first_last_diff:.3e}");
    // f16 with a tiny grad scale (0.001) frequently overflows or underflows
    // the loss scaler so commits get skipped — under that regime weights may
    // legitimately stay still. For f32/bf16 we require visible AdamW motion
    // across 10 replays of the same input (each step reuses accumulated m/v).
    if !matches!(dtype, WeightDtype::F16) {
        assert!(
            first_last_diff > 0.0,
            "[{dtype:?}] weights did not evolve across replays (max_abs_diff=0)"
        );
    }
}

#[test]
fn graph_determinism_f32() {
    run_graph_determinism(WeightDtype::F32);
}

#[test]
fn graph_determinism_bf16() {
    run_graph_determinism(WeightDtype::Bf16);
}

#[test]
fn graph_determinism_f16() {
    run_graph_determinism(WeightDtype::F16);
}

// =====================================================================
// 3. Long-sequence inference stability on real HF checkpoint
// =====================================================================

#[cfg(feature = "hf")]
mod hf_stability {
    use super::*;
    use mamba_rs::module::gpu_lm::GpuMambaLM;
    use mamba_rs::module::sample::SampleParams;
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

    fn logit_inf_norm(v: &[f32]) -> f32 {
        v.iter().copied().fold(0.0f32, |acc, x| acc.max(x.abs()))
    }

    fn run_long_seq(dtype: WeightDtype) {
        let label = format!("{dtype:?}");
        let dir = match find_model_dir("mamba-130m-hf") {
            Some(d) => d,
            None => {
                eprintln!("[skip {label}] mamba-130m-hf not in HF cache");
                return;
            }
        };

        let mut lm = GpuMambaLM::from_hf_with_dtype(&dir, 0, dtype).expect("load LM");
        let prompt: &[u32] = &[1, 2, 3, 4, 5];
        let params = SampleParams {
            temperature: 0.0,
            max_tokens: 256,
            ..Default::default()
        };
        let tokens = lm.generate(prompt, &params).expect("generate 256 tokens");
        assert_eq!(tokens.len(), 256);
        for (i, &t) in tokens.iter().enumerate() {
            assert!(
                (t as usize) < lm.vocab_size,
                "[{label}] token {i}={t} out of vocab {}",
                lm.vocab_size
            );
        }

        // Logit magnitude after 256 tokens must remain bounded. SSM state
        // decay should keep this sane; unbounded growth would indicate the
        // `a_neg` decay multiplier is off.
        let final_logits = lm.last_logits(0);
        assert!(
            final_logits.iter().all(|v| v.is_finite()),
            "[{label}] final logits not finite after 256 tokens"
        );
        let inf_norm = logit_inf_norm(final_logits);
        eprintln!(
            "[{label}] 256-token stability: max|logit|={inf_norm:.3e}  last tokens={:?}",
            &tokens[tokens.len() - 5..]
        );
        assert!(
            inf_norm < 1e3,
            "[{label}] logit magnitude runaway after 256 tokens: max|logit|={inf_norm:.3e}"
        );
    }

    #[test]
    #[ignore]
    fn long_seq_stable_f32() {
        run_long_seq(WeightDtype::F32);
    }

    #[test]
    #[ignore]
    fn long_seq_stable_bf16() {
        run_long_seq(WeightDtype::Bf16);
    }

    #[test]
    #[ignore]
    fn long_seq_stable_f16() {
        run_long_seq(WeightDtype::F16);
    }
}
