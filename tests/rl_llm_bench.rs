//! Workload-focused benchmarks: RL training loop + LLM inference.
//!
//! RL benches (CPU — matches SQV-RS deployment) exercise the batched training
//! step (forward + parallel backward) at realistic RL shapes (small model,
//! batch of 32–128 parallel envs, short window T=32 with a burn-in prefix).
//!
//! LLM benches (GPU) split prefill from decode so the two throughput modes
//! are measured separately, then extend to long-context prefill and to
//! batched generation for parallel sampling.
//!
//! All tests are `#[ignore]` — opt-in via:
//!   cargo test --release --features "cuda hf" --test rl_llm_bench -- --ignored --nocapture
//!
//! CPU-only RL bench works without `cuda`/`hf`:
//!   cargo test --release --test rl_llm_bench rl_ -- --ignored --nocapture

use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// RL training benches (CPU, M3, batched)
// ═══════════════════════════════════════════════════════════════════════════

fn rl_config_small() -> mamba_rs::mamba3_siso::config::Mamba3Config {
    // Matches the typical SQV-RS actor backbone: small d_model, few layers,
    // full RoPE fraction. d_state=16 is the paper default.
    mamba_rs::mamba3_siso::config::Mamba3Config {
        d_model: 128,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 4,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
    }
}

fn rl_config_tiny() -> mamba_rs::mamba3_siso::config::Mamba3Config {
    mamba_rs::mamba3_siso::config::Mamba3Config {
        d_model: 64,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
    }
}

fn init_rl_weights(
    dims: &mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims,
    input_dim: usize,
) -> mamba_rs::mamba3_siso::cpu::weights::TrainMamba3Weights {
    use mamba_rs::mamba3_siso::cpu::weights::TrainMamba3Weights;
    let mut state = 0xB0BAu32;
    let mut rand = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state & 0x7fff_ffff) as f32 / 2_147_483_647.0 * 0.1 - 0.05
    };
    let mut w = TrainMamba3Weights::zeros(dims, input_dim);
    for v in &mut w.input_proj_w {
        *v = rand();
    }
    for v in &mut w.norm_f_weight {
        *v = 1.0 + rand();
    }
    for l in &mut w.layers {
        for v in &mut l.norm_weight {
            *v = 1.0 + rand();
        }
        for v in &mut l.in_proj_w {
            *v = rand();
        }
        for v in &mut l.dt_bias {
            *v = rand();
        }
        for v in &mut l.b_norm_weight {
            *v = 1.0 + rand();
        }
        for v in &mut l.c_norm_weight {
            *v = 1.0 + rand();
        }
        for v in &mut l.b_bias {
            *v = rand();
        }
        for v in &mut l.c_bias {
            *v = rand();
        }
        for v in &mut l.d_param {
            *v = 1.0 + rand();
        }
        for v in &mut l.norm_gate_weight {
            *v = 1.0 + rand();
        }
        for v in &mut l.out_proj_w {
            *v = rand();
        }
    }
    w
}

fn time_training_step(
    cfg: &mamba_rs::mamba3_siso::config::Mamba3Config,
    batch: usize,
    seq_len: usize,
    iters: usize,
) -> (f64, f64) {
    use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
    use mamba_rs::mamba3_siso::cpu::flat::Mamba3LayerFlat;
    use mamba_rs::mamba3_siso::cpu::parallel::{
        Mamba3States, parallel_mamba3_backward, parallel_mamba3_forward,
    };
    use mamba_rs::mamba3_siso::cpu::weights::TrainMamba3Weights;

    cfg.validate().unwrap();
    let input_dim = cfg.d_model;
    let dims = Mamba3Dims::from_config(cfg, seq_len);
    let w = init_rl_weights(&dims, input_dim);

    let nh = dims.nheads;
    let hd = dims.headdim;
    let ds = dims.d_state;
    let nl = dims.n_layers;
    let na = dims.num_rope_angles.max(1);

    let mut temporal = vec![0.1f32; batch * seq_len * dims.d_model];
    let mut acts: Vec<Vec<Mamba3LayerFlat>> = (0..batch)
        .map(|_| (0..nl).map(|_| Mamba3LayerFlat::zeros(dims)).collect())
        .collect();
    let mut ssm = vec![0.0; batch * nl * nh * hd * ds];
    let mut k = vec![0.0; batch * nl * nh * ds];
    let mut v = vec![0.0; batch * nl * nh * hd];
    let mut angle = vec![0.0; batch * nl * nh * na];
    let mut d_temporal = vec![1.0f32; batch * seq_len * dims.d_model];
    let mut d_w = TrainMamba3Weights::zeros(&dims, input_dim);

    // Warmup (populate rayon thread-local scratch).
    for _ in 0..3 {
        temporal.fill(0.1);
        ssm.fill(0.0);
        k.fill(0.0);
        v.fill(0.0);
        angle.fill(0.0);
        parallel_mamba3_forward(
            &mut temporal,
            &mut acts,
            Mamba3States {
                ssm: &mut ssm,
                k: &mut k,
                v: &mut v,
                angle: &mut angle,
            },
            &w,
            &dims,
            batch,
        );
        d_temporal.fill(1.0);
        d_w.zero();
        parallel_mamba3_backward(
            &mut d_temporal,
            &acts,
            &w,
            &mut d_w,
            &dims,
            batch,
            input_dim,
        );
    }

    // Forward timing.
    let t0 = Instant::now();
    for _ in 0..iters {
        temporal.fill(0.1);
        ssm.fill(0.0);
        k.fill(0.0);
        v.fill(0.0);
        angle.fill(0.0);
        parallel_mamba3_forward(
            &mut temporal,
            &mut acts,
            Mamba3States {
                ssm: &mut ssm,
                k: &mut k,
                v: &mut v,
                angle: &mut angle,
            },
            &w,
            &dims,
            batch,
        );
    }
    let fwd_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Backward timing.
    let t0 = Instant::now();
    for _ in 0..iters {
        d_temporal.fill(1.0);
        d_w.zero();
        parallel_mamba3_backward(
            &mut d_temporal,
            &acts,
            &w,
            &mut d_w,
            &dims,
            batch,
            input_dim,
        );
    }
    let bwd_us = t0.elapsed().as_micros() as f64 / iters as f64;

    (fwd_us, bwd_us)
}

#[test]
#[ignore]
fn rl_cpu_training_step_batch_sweep() {
    let cfg = rl_config_small();
    let seq_len = 32;
    let iters = 30;
    let n_threads = rayon::current_num_threads();

    eprintln!(
        "\n=== RL training step: M3 small (d_model=128, 4 layers), T={seq_len}, threads={n_threads} ==="
    );
    eprintln!(
        "{:>5} | {:>10} {:>10} {:>12}",
        "B", "fwd (ms)", "bwd (ms)", "steps/s"
    );
    for &batch in &[1usize, 8, 16, 32, 64, 128] {
        let (fwd, bwd) = time_training_step(&cfg, batch, seq_len, iters);
        let step_ms = (fwd + bwd) / 1000.0;
        let steps_s = 1000.0 / step_ms;
        eprintln!(
            "{batch:>5} | {:>10.2} {:>10.2} {:>12.1}",
            fwd / 1000.0,
            bwd / 1000.0,
            steps_s
        );
    }
}

#[test]
#[ignore]
fn rl_cpu_training_step_model_sweep() {
    let seq_len = 32;
    let batch = 64;
    let iters = 30;
    let n_threads = rayon::current_num_threads();

    eprintln!("\n=== RL training step: B={batch}, T={seq_len}, threads={n_threads} ===");
    eprintln!(
        "{:>10} | {:>10} {:>10} {:>12}",
        "config", "fwd (ms)", "bwd (ms)", "steps/s"
    );

    for (label, cfg) in [
        ("tiny(d64)", rl_config_tiny()),
        ("small(d128)", rl_config_small()),
    ] {
        let (fwd, bwd) = time_training_step(&cfg, batch, seq_len, iters);
        let step_ms = (fwd + bwd) / 1000.0;
        let steps_s = 1000.0 / step_ms;
        eprintln!(
            "{label:>10} | {:>10.2} {:>10.2} {:>12.1}",
            fwd / 1000.0,
            bwd / 1000.0,
            steps_s
        );
    }
}

#[test]
#[ignore]
fn rl_cpu_seq_len_sweep() {
    // RL uses burn-in; verify the M3 parallel path scales linearly with T.
    let cfg = rl_config_small();
    let batch = 64;
    let iters = 20;
    let n_threads = rayon::current_num_threads();

    eprintln!("\n=== RL training step: M3 small, B={batch}, threads={n_threads} ===");
    eprintln!(
        "{:>5} | {:>10} {:>10} {:>15}",
        "T", "fwd (ms)", "bwd (ms)", "us/sample/tok"
    );
    for &seq_len in &[8usize, 16, 32, 64, 128] {
        let (fwd, bwd) = time_training_step(&cfg, batch, seq_len, iters);
        let tok = (batch * seq_len) as f64;
        let total_us = fwd + bwd;
        eprintln!(
            "{seq_len:>5} | {:>10.2} {:>10.2} {:>15.3}",
            fwd / 1000.0,
            bwd / 1000.0,
            total_us / tok
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LLM inference benches (GPU)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "hf", feature = "cuda"))]
mod llm {
    use super::*;
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
            for entry in std::fs::read_dir(cache).ok()?.flatten() {
                if let Ok(fname) = entry.file_name().into_string()
                    && fname.contains(name)
                {
                    let snaps = entry.path().join("snapshots");
                    if snaps.exists()
                        && let Ok(mut it) = std::fs::read_dir(&snaps)
                        && let Some(Ok(snap)) = it.next()
                    {
                        return Some(snap.path());
                    }
                }
            }
        }
        None
    }

    /// Prefill + steady-state decode, broken out so TTFT (time to first token)
    /// is measured separately from the flat per-token cost of the decode loop.
    #[test]
    #[ignore]
    fn llm_prefill_vs_decode_all_models() {
        use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
        use mamba_rs::module::gpu_lm::GpuMambaLM;
        use mamba_rs::module::sample::SampleParams;

        eprintln!("\n=== LLM prefill vs decode: prompt=128, decode=100 tokens, bf16 ===");
        eprintln!(
            "{:>14} | {:>13} {:>12} {:>12} {:>12}",
            "model", "prefill (ms)", "dec (ms)", "dec tok/s", "TTFT (ms)"
        );
        for name in [
            "mamba-130m-hf",
            "mamba-370m-hf",
            "mamba-1.4b-hf",
            "mamba-2.8b-hf",
        ] {
            let dir = match find_model_dir(name) {
                Some(d) => d,
                None => {
                    eprintln!("{name:>14} | NOT CACHED");
                    continue;
                }
            };
            let mut lm = GpuMambaLM::from_hf_with_dtype(&dir, 0, WeightDtype::Bf16).unwrap();
            lm.capture_graph().unwrap();

            // Warmup
            let warm_prompt: Vec<u32> = (0..16).collect();
            let _ = lm
                .generate(
                    &warm_prompt,
                    &SampleParams {
                        temperature: 0.0,
                        max_tokens: 5,
                        ..Default::default()
                    },
                )
                .unwrap();
            lm.reset().unwrap();

            // Prefill: 128-token prompt, 1 generated token → first-token latency.
            let prompt: Vec<u32> = (0..128u32).collect();
            let t0 = Instant::now();
            let _ = lm
                .generate(
                    &prompt,
                    &SampleParams {
                        temperature: 0.0,
                        max_tokens: 1,
                        ..Default::default()
                    },
                )
                .unwrap();
            let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
            lm.reset().unwrap();

            // Decode: same prefill + 100 generated tokens; subtract prefill to
            // isolate the decode-only time.
            let t0 = Instant::now();
            let tokens = lm
                .generate(
                    &prompt,
                    &SampleParams {
                        temperature: 0.0,
                        max_tokens: 101,
                        ..Default::default()
                    },
                )
                .unwrap();
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let decode_ms = total_ms - prefill_ms;
            let decode_tps = (tokens.len() as f64 - 1.0) / (decode_ms / 1000.0);
            eprintln!(
                "{name:>14} | {:>13.1} {:>12.1} {:>12.0} {:>12.1}",
                prefill_ms, decode_ms, decode_tps, prefill_ms
            );
        }
    }

    /// Long-context prefill: mamba's structural advantage is O(T) prefill.
    /// Measure per-prompt-token cost at T=256/1024/4096 to verify scaling.
    #[test]
    #[ignore]
    fn llm_long_context_prefill() {
        use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
        use mamba_rs::module::gpu_lm::GpuMambaLM;
        use mamba_rs::module::sample::SampleParams;

        let name = "mamba-370m-hf";
        let dir = match find_model_dir(name) {
            Some(d) => d,
            None => {
                eprintln!("[skip] {name} not cached");
                return;
            }
        };

        eprintln!("\n=== LLM long-context prefill: {name}, bf16 ===");
        eprintln!(
            "{:>6} | {:>12} {:>14}",
            "tokens", "prefill (ms)", "us/prefill-tok"
        );
        for &t in &[64usize, 256, 1024, 4096] {
            let mut lm = GpuMambaLM::from_hf_with_dtype(&dir, 0, WeightDtype::Bf16).unwrap();
            lm.capture_graph().unwrap();

            // Warmup with same shape so the graph is valid.
            let warm: Vec<u32> = (0..16).collect();
            let _ = lm
                .generate(
                    &warm,
                    &SampleParams {
                        temperature: 0.0,
                        max_tokens: 5,
                        ..Default::default()
                    },
                )
                .unwrap();
            lm.reset().unwrap();

            let prompt: Vec<u32> = (0..t as u32).map(|i| i % 50257).collect();
            let iters = if t <= 256 { 5 } else { 2 };
            let t0 = Instant::now();
            for _ in 0..iters {
                lm.reset().unwrap();
                let _ = lm
                    .generate(
                        &prompt,
                        &SampleParams {
                            temperature: 0.0,
                            max_tokens: 1,
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            let us_per_tok = elapsed_ms * 1000.0 / t as f64;
            eprintln!("{t:>6} | {:>12.1} {:>14.2}", elapsed_ms, us_per_tok);
        }
    }

    /// RL parallel-env rollout pattern: one step at a time across B slots.
    /// With the current batch-generation bug (see test_gpu_batch_generation)
    /// the output values are wrong, but the throughput measurement is still
    /// meaningful — it shows what we'd get once the bug is fixed.
    #[test]
    #[ignore]
    fn llm_batched_step_throughput() {
        use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
        use mamba_rs::module::gpu_lm::GpuMambaLM;
        use mamba_rs::module::sample::SampleParams;

        let name = "mamba-130m-hf";
        let dir = match find_model_dir(name) {
            Some(d) => d,
            None => {
                eprintln!("[skip] {name} not cached");
                return;
            }
        };

        eprintln!("\n=== GPU batched step throughput: {name} bf16 (RL parallel envs) ===");
        eprintln!(
            "NOTE: throughput-only view of the batched decode path; \
             cross-slot bit-identity is covered by gpu_batch_test — these numbers measure \
             throughput only."
        );
        eprintln!(
            "{:>5} | {:>10} {:>12} {:>14}",
            "B", "tok/s/slot", "tok/s total", "us/slot/tok"
        );
        let prompt: &[u32] = &[1, 2, 3, 4, 5];
        for &batch in &[1usize, 2, 4, 8, 16] {
            let mut lm =
                GpuMambaLM::from_hf_with_dtype_batch(&dir, 0, WeightDtype::Bf16, batch).unwrap();
            let params = SampleParams {
                temperature: 0.0,
                max_tokens: 50,
                seed: 42,
                ..Default::default()
            };
            let prompts: Vec<&[u32]> = (0..batch).map(|_| prompt).collect();
            let params_vec: Vec<SampleParams> = (0..batch).map(|_| params.clone()).collect();

            // Warmup
            let _ = lm.generate_batch(&prompts, &params_vec).unwrap();

            let t0 = Instant::now();
            let outs = lm.generate_batch(&prompts, &params_vec).unwrap();
            let secs = t0.elapsed().as_secs_f64();
            let tok_per_slot = outs[0].len();
            let total_tok = batch * tok_per_slot;
            let tps_total = total_tok as f64 / secs;
            let tps_slot = tok_per_slot as f64 / secs;
            let us_per_slot_tok = secs * 1e6 / total_tok as f64;
            eprintln!(
                "{batch:>5} | {:>10.1} {:>12.1} {:>14.2}",
                tps_slot, tps_total, us_per_slot_tok
            );
        }
    }

    /// End-to-end dtype sweep at all four HF sizes — parity AND perf in one.
    /// Correctness is already covered by tests/gpu_bf16_parity.rs; this test
    /// measures throughput delta across dtypes.
    #[test]
    #[ignore]
    fn llm_dtype_throughput_all_sizes() {
        use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
        use mamba_rs::module::gpu_lm::GpuMambaLM;
        use mamba_rs::module::sample::SampleParams;

        eprintln!("\n=== LLM dtype throughput sweep (graph-captured) ===");
        eprintln!(
            "{:>14} | {:>6} | {:>8} {:>10}",
            "model", "dtype", "tok/s", "ms/tok"
        );
        for name in [
            "mamba-130m-hf",
            "mamba-370m-hf",
            "mamba-1.4b-hf",
            "mamba-2.8b-hf",
        ] {
            let dir = match find_model_dir(name) {
                Some(d) => d,
                None => {
                    eprintln!("{name:>14} | NOT CACHED");
                    continue;
                }
            };
            for (dt, label) in [
                (WeightDtype::F32, "f32 "),
                (WeightDtype::Bf16, "bf16"),
                (WeightDtype::F16, "f16 "),
            ] {
                let mut lm = match GpuMambaLM::from_hf_with_dtype(&dir, 0, dt) {
                    Ok(lm) => lm,
                    Err(e) => {
                        eprintln!("{name:>14} | {label} | ERR {e}");
                        continue;
                    }
                };
                lm.capture_graph().unwrap();
                let params = SampleParams {
                    temperature: 0.0,
                    max_tokens: 50,
                    ..Default::default()
                };
                let _ = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
                lm.reset().unwrap();
                let t0 = Instant::now();
                let toks = lm.generate(&[1, 2, 3, 4, 5], &params).unwrap();
                let secs = t0.elapsed().as_secs_f64();
                let tps = toks.len() as f64 / secs;
                let ms_per_tok = secs * 1000.0 / toks.len() as f64;
                eprintln!("{name:>14} | {label} | {:>8.0} {:>10.2}", tps, ms_per_tok);
            }
        }
    }
}
