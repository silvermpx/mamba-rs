//! Trainer step wall clock with the deterministic GEMM triad against the
//! vendor paths, across small to 770m-class shapes and f32 / bf16 / f16.
//! Timing only: the determinism contract tests live in
//! tests/gemm_bi_determinism.rs.
#![cfg(feature = "cuda")]

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
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

fn cfg() -> MambaConfig {
    MambaConfig {
        d_model: 128,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    }
}

fn bench_gemm_bi_vs_tf32() {
    use std::time::Instant;
    // (d_model, n_layers, batch, seq_len, label)
    let shapes = [
        (128usize, 2usize, 16usize, 64usize, "RL-small d128"),
        (256, 4, 16, 128, "d256"),
        (768, 4, 8, 256, "130m-ish d768"),
        (1536, 2, 4, 256, "770m-ish d1536"),
    ];
    for (dm, nl, b, t, label) in shapes {
        let cfg = MambaConfig {
            d_model: dm,
            n_layers: nl,
            ..cfg()
        };
        let input_dim = cfg.d_model;
        let session = TrainSessionCfg {
            input_dim,
            batch: b,
            seq_len: t,
            lr: 1e-4,
            weight_decay: 0.0,
        };
        let n = b * t * input_dim;
        let input = det(n, 1, 1.0);
        let dtemp = det(n, 2, 0.1);

        let time_mode = |invariant: bool, tc: bool, dtype: WeightDtype| -> f64 {
            let mut cpu = MambaWeights::init(&cfg, input_dim, 7);
            if dtype != WeightDtype::F32 {
                // The mixed-precision pipeline requires an identity input_proj.
                cpu.input_proj_w.clear();
                cpu.input_proj_b.clear();
            }
            for lw in cpu.layers.iter_mut() {
                lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
            }
            let mut tr = MambaTrainer::new_full(0, &cpu, cfg, session, dtype).expect("trainer");
            tr.ctx()
                .set_gemm_mode(if invariant {
                    GemmMode::Deterministic
                } else {
                    GemmMode::CublasFast
                })
                .unwrap();
            tr.ctx().set_bi_tensor_cores(tc);
            for _ in 0..3 {
                tr.step(&input, &dtemp).expect("warmup");
            }
            let iters = 20;
            let start = Instant::now();
            for _ in 0..iters {
                tr.step(&input, &dtemp).expect("step");
            }
            start.elapsed().as_secs_f64() / iters as f64
        };

        for dt in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
            // flag-off baseline: cuBLAS TF32 for f32, cuBLAS GemmEx
            // PEDANTIC (f32 accumulate, no tensor cores) for bf16/f16.
            let baseline = if dt == WeightDtype::F32 {
                "cuBLAS-TF32"
            } else {
                "cuBLAS-PEDANTIC"
            };
            let t_blas = time_mode(false, false, dt);
            let t_bi = time_mode(true, false, dt);
            eprintln!(
                "[{label} {dt:?}] B={b} T={t}: {baseline} {:.3} ms/step | gemm_bi {:.3} ms/step | ratio {:.2}x",
                t_blas * 1e3,
                t_bi * 1e3,
                t_bi / t_blas
            );
            if dt != WeightDtype::F32 {
                let t_tc = time_mode(true, true, dt);
                eprintln!(
                    "[{label} {dt:?}] B={b} T={t}: gemm_bi+TC {:.3} ms/step | vs {baseline} {:.2}x | vs scalar bi {:.2}x",
                    t_tc * 1e3,
                    t_tc / t_blas,
                    t_tc / t_bi
                );
            }
        }
    }
}

// Run every instrument, or only the ones named on the command line:
// `cargo bench --bench <target> --features cuda -- <name> [<name> ...]`.
fn main() {
    let selected: Vec<String> = std::env::args().skip(1).collect();
    let run = |name: &str| selected.is_empty() || selected.iter().any(|s| s == name);
    if run("bench_gemm_bi_vs_tf32") {
        bench_gemm_bi_vs_tf32();
    }
}
