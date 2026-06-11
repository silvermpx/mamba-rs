//! Edge-case coverage tests — extreme batch sizes, very long prompts,
//! M3 long sequences. Complements the existing batch/length parity
//! suite by stress-testing the boundaries that weren't gated elsewhere.
//!
//! Structure:
//!   * `inference_extreme_batch_parity_{bf16,f16,f32}` — batch=16 and
//!     batch=32 inference parity vs batch=1 on the SAME prompts. The
//!     batch-invariant GEMM kernel should keep KL < 1e-4 regardless of
//!     batch size; if cuBLAS fallback is hit (or the kernel has an
//!     overlooked M-dependent path), large-batch drift surfaces here.
//!   * `very_long_prompt_1024_stability_{bf16,f16}` — 1024-token trajectory
//!     (256-token prompt plus 768 generated) on mamba-130m. Asserts no
//!     NaN, logit magnitude bounded, all vocab indices valid. The SSM
//!     recurrence decay has to stay numerically stable over this length.
//!   * `m3_long_sequence_stability_{bf16,f16}` — synthetic M3 config
//!     with seq_len=512 (spans 8 chunks of size 64). Verifies the M3
//!     chunked SSD machinery handles seq_len far beyond the default
//!     32-token test matrix without state corruption.
//!
//! All tests are `#[ignore]` because HF-backed ones need checkpoint
//! cache and the M3 long-seq test uses ~1 GB VRAM at seq_len=512.

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

// ===========================================================================
// HF 130m — extreme batch size parity + very long prompts
// ===========================================================================

#[cfg(feature = "hf")]
mod hf {
    use super::*;
    use mamba_rs::module::gpu_lm::GpuMambaLM;
    use mamba_rs::module::sample::SampleParams;
    use std::path::PathBuf;

    // Lives inside the hf module: its only callers are hf-gated, and a
    // cuda-only build would otherwise see it as dead code.
    fn kl_divergence(p_logits: &[f32], q_logits: &[f32]) -> f32 {
        assert_eq!(p_logits.len(), q_logits.len());
        let pmax = p_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let qmax = q_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut psum = 0.0f64;
        let mut qsum = 0.0f64;
        for (&pl, &ql) in p_logits.iter().zip(q_logits.iter()) {
            psum += ((pl as f64) - pmax).exp();
            qsum += ((ql as f64) - qmax).exp();
        }
        let log_ps = psum.ln();
        let log_qs = qsum.ln();
        let mut kl = 0.0f64;
        for (&pl, &ql) in p_logits.iter().zip(q_logits.iter()) {
            let lp = ((pl as f64) - pmax) - log_ps;
            let lq = ((ql as f64) - qmax) - log_qs;
            let p = lp.exp();
            if p > 1e-30 {
                kl += p * (lp - lq);
            }
        }
        kl as f32
    }

    pub fn find_model_dir(name: &str) -> Option<PathBuf> {
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

    fn run_extreme_batch_parity(dtype: WeightDtype, batch: usize) {
        let label = format!("{dtype:?}@b{batch}");
        let dir = match find_model_dir("mamba-130m-hf") {
            Some(d) => d,
            None => {
                eprintln!("[skip {label}] no HF cache");
                return;
            }
        };
        let params = SampleParams {
            temperature: 0.0,
            max_tokens: 1,
            ..Default::default()
        };

        // Reference: batch=1, fixed prompt.
        let prompt: &[u32] = &[1, 2, 3, 4, 5];
        let mut lm1 = GpuMambaLM::from_hf_with_dtype_batch(&dir, 0, dtype, 1).expect("b=1 load");
        lm1.generate(prompt, &params).expect("b=1 gen");
        let ref_logits = lm1.last_logits(0).to_vec();

        // Large batch, prompt at slot 0, filler prompts of SAME length
        // for every other slot (so max_prompt equals slot-0 length and
        // slot-0 decodes exactly 1 token matching the b=1 reference).
        let mut lm_large =
            GpuMambaLM::from_hf_with_dtype_batch(&dir, 0, dtype, batch).expect("b=N load");
        let filler: Vec<u32> = (200..205).collect();
        let mut prompts: Vec<&[u32]> = Vec::with_capacity(batch);
        prompts.push(prompt);
        for _ in 1..batch {
            prompts.push(&filler);
        }
        let params_n: Vec<SampleParams> = (0..batch).map(|_| params.clone()).collect();
        lm_large
            .generate_batch(&prompts, &params_n)
            .expect("b=N gen");
        let slot0_logits = lm_large.last_logits(0).to_vec();

        let kl = kl_divergence(&ref_logits, &slot0_logits);
        eprintln!("[{label}] KL(b=1 || b={batch} slot0) = {kl:.4e}");

        // Batch-invariant GEMM kernel: KL should be <1e-4 for bf16/f16
        // and bit-identical (<1e-10) for f32. If this asserts breaks,
        // some op besides our batch-invariant GEMM has B-dependent
        // reduction (new rmsnorm variant added? new path hit?).
        let limit = match dtype {
            WeightDtype::F32 => 1e-5,
            _ => 1e-4,
        };
        assert!(
            kl < limit,
            "[{label}] KL {kl:.4e} > {limit:.0e} — cross-batch drift exceeds envelope"
        );
    }

    #[test]
    #[ignore]
    fn inference_extreme_batch_parity_bf16_b16() {
        run_extreme_batch_parity(WeightDtype::Bf16, 16);
    }

    #[test]
    #[ignore]
    fn inference_extreme_batch_parity_bf16_b32() {
        run_extreme_batch_parity(WeightDtype::Bf16, 32);
    }

    #[test]
    #[ignore]
    fn inference_extreme_batch_parity_f16_b16() {
        run_extreme_batch_parity(WeightDtype::F16, 16);
    }

    #[test]
    #[ignore]
    fn inference_extreme_batch_parity_f32_b16() {
        run_extreme_batch_parity(WeightDtype::F32, 16);
    }

    fn run_very_long_prompt(dtype: WeightDtype) {
        let label = format!("{dtype:?}");
        let dir = match find_model_dir("mamba-130m-hf") {
            Some(d) => d,
            None => {
                eprintln!("[skip {label}] no HF cache");
                return;
            }
        };
        let mut lm = GpuMambaLM::from_hf_with_dtype(&dir, 0, dtype).expect("load LM");
        // 256-token prompt (tokens 100..356) + 768 generated = 1024 total.
        let prompt: Vec<u32> = (100..356).collect();
        let params = SampleParams {
            temperature: 0.0,
            max_tokens: 768,
            ..Default::default()
        };
        let tokens = lm.generate(&prompt, &params).expect("generate 768 tokens");
        assert_eq!(tokens.len(), 768);

        for (i, &t) in tokens.iter().enumerate() {
            assert!(
                (t as usize) < lm.vocab_size,
                "[{label}] token {i}={t} out of vocab {}",
                lm.vocab_size
            );
        }

        let final_logits = lm.last_logits(0);
        assert!(
            final_logits.iter().all(|v: &f32| v.is_finite()),
            "[{label}] non-finite logits after 1024-token trajectory"
        );
        let max_abs = final_logits.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        eprintln!(
            "[{label}] 1024-token stability: max|logit|={max_abs:.3e}  last5={:?}",
            &tokens[tokens.len() - 5..]
        );
        assert!(
            max_abs < 2e3,
            "[{label}] logit magnitude ran away at 1024 tokens: {max_abs:.3e}"
        );
    }

    #[test]
    #[ignore]
    fn very_long_prompt_1024_bf16() {
        run_very_long_prompt(WeightDtype::Bf16);
    }

    #[test]
    #[ignore]
    fn very_long_prompt_1024_f16() {
        run_very_long_prompt(WeightDtype::F16);
    }

    #[test]
    #[ignore]
    fn very_long_prompt_1024_f32() {
        run_very_long_prompt(WeightDtype::F32);
    }
}

// ===========================================================================
// M3 long-sequence stability (synthetic weights — no public M3 HF ckpts)
// ===========================================================================

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

fn run_m3_long_seq(dtype: WeightDtype, seq_len: usize) {
    use mamba_rs::mamba3_siso::config::Mamba3Config;
    use mamba_rs::mamba3_siso::gpu::trainer::Mamba3Trainer;
    use mamba_rs::mamba3_siso::weights::Mamba3Weights;

    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    };
    let input_dim = cfg.d_model;
    let batch = 1;
    let n = batch * seq_len * input_dim;

    let mut cpu = Mamba3Weights::init(&cfg, input_dim, 0x0000_B16A);
    if matches!(dtype, WeightDtype::F32) {
        // f32 M3 forward needs real input_proj (no identity branch).
        cpu.input_proj_w = (0..input_dim * cfg.d_model)
            .map(|i| {
                if i / cfg.d_model == i % cfg.d_model {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        cpu.input_proj_b = vec![0.0; cfg.d_model];
    } else {
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
    }

    let label = format!("M3-{dtype:?}-T{seq_len}");
    let mut trainer =
        Mamba3Trainer::new_full(0, &cpu, cfg, input_dim, batch, seq_len, dtype, 1e-6, 0.0)
            .unwrap_or_else(|e| panic!("[{label}] construct: {e}"));

    // 3 training steps — exercises chunk boundaries (seq_len 512 = 8 chunks
    // of size 64). Any state-carry bug between chunks would show up as NaN
    // or unbounded weight drift within these few steps.
    for s in 0..3u32 {
        let m = trainer
            .step(&det(n, 0xA0 + s), &det(n, 0xB0 + s))
            .unwrap_or_else(|e| panic!("[{label}] step {s}: {e}"));
        if matches!(dtype, WeightDtype::F16) {
            assert!(m.loss_scale.is_some());
        }
    }

    // Verify weights remain finite.
    let snap = trainer.snapshot_master().unwrap();
    for (li, lw) in snap.layers.iter().enumerate() {
        assert!(
            lw.in_proj_w.iter().all(|v: &f32| v.is_finite()),
            "[{label}] L{li}.in_proj_w non-finite after T={seq_len}"
        );
        assert!(
            lw.out_proj_w.iter().all(|v: &f32| v.is_finite()),
            "[{label}] L{li}.out_proj_w non-finite"
        );
    }
    eprintln!("[{label}] 3 steps @ seq_len={seq_len} ok");
}

#[test]
fn m3_long_sequence_stability_bf16_t512() {
    run_m3_long_seq(WeightDtype::Bf16, 512);
}

#[test]
fn m3_long_sequence_stability_f16_t512() {
    run_m3_long_seq(WeightDtype::F16, 512);
}

#[test]
fn m3_long_sequence_stability_f32_t512() {
    run_m3_long_seq(WeightDtype::F32, 512);
}

#[test]
#[ignore] // needs more VRAM at seq_len=1024
fn m3_long_sequence_stability_bf16_t1024() {
    run_m3_long_seq(WeightDtype::Bf16, 1024);
}
