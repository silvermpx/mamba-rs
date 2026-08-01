//! Manual (--ignored): full training-step pace, Sequential vs Auto scan,
//! at the Prism classifier shape (d384x24, T=4621, input_dim=1024, B=1,
//! bf16 — the production trainer dtype). B=1 so the bench fits BESIDE a
//! live B=2 training run on one card (~9.6 GiB vs ~14 GiB headroom); the
//! seq/parallel RATIO transfers to B=2.

#![cfg(feature = "cuda")]

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::trainer::{BackwardOpts, MambaTrainer};
use mamba_rs::weights::MambaWeights;

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
fn scan_mode_bench_classifier_shape() {
    let (batch, seq_len, input_dim) = (1usize, 4621usize, 1024usize);
    // Third arm: Auto + the opt-in non-PEDANTIC GEMM compute (fast_gemm) —
    // measures the tensor-core headroom of the typed training GEMMs.
    for (mode, fast) in [
        (ScanMode::Sequential, false),
        (ScanMode::Auto, false),
        (ScanMode::Auto, true),
    ] {
        let cfg = MambaConfig {
            d_model: 384,
            n_layers: 24,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            scan_mode: mode,
            rms_norm_eps: 1e-5,
        };
        let mut cpu = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
        for lw in cpu.layers.iter_mut() {
            lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
        }
        let mut trainer = MambaTrainer::new_with_dtype(
            0,
            &cpu,
            cfg,
            input_dim,
            batch,
            seq_len,
            WeightDtype::Bf16,
        )
        .expect("trainer");
        trainer.ctx().set_fast_gemm(fast);
        let input = det(batch * seq_len * input_dim, 0xA5, 0.05);
        let mut temporal = vec![0.0f32; batch * seq_len * cfg.d_model];
        let d_temporal = det(batch * seq_len * cfg.d_model, 0xB6, 0.01);
        // Warm-up (kernel compile + caches), then timed steps.
        for _ in 0..2 {
            trainer.forward(&input, &mut temporal).expect("fwd");
            trainer
                .backward_step(&d_temporal, BackwardOpts::default())
                .expect("bwd");
        }
        let reps = 8;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            trainer.forward(&input, &mut temporal).expect("fwd");
            trainer
                .backward_step(&d_temporal, BackwardOpts::default())
                .expect("bwd");
        }
        let s_step = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "BENCH {mode:?} fast_gemm={fast}: {s_step:.3} s/step (fwd+bwd, B={batch} T={seq_len} bf16)"
        );
    }
}
