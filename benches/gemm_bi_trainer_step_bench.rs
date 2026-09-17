//! Trainer step wall clock with the deterministic GEMM triad against the
//! vendor paths, across small to 770m-class shapes and f32 / bf16 / f16.
//! Timing only: the determinism contract tests live in
//! tests/gemm_bi_determinism.rs.
#![cfg(feature = "cuda")]

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::context::{F32TriadPolicy, HalfTriadPolicy};
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

        // `tf32` opts the f32 products into the deterministic TF32 kernels where
        // a measured route exists; `streamk` opts the half weight gradient into
        // the measured stream-K kernels. Both are 0.7.0 settings with no
        // counterpart in the previous release.
        let time_mode =
            |mode: GemmMode, tc: bool, tf32: bool, streamk: bool, dtype: WeightDtype| -> f64 {
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
                tr.ctx().set_gemm_mode(mode).unwrap();
                tr.ctx().route_controls().set_tensor_cores(tc);
                if mode == GemmMode::Deterministic {
                    tr.ctx().route_controls().set_f32_policy(if tf32 {
                        F32TriadPolicy::AllowDeterministicTf32
                    } else {
                        F32TriadPolicy::ExactScalarFma
                    });
                    tr.ctx().route_controls().set_half_policy(if streamk {
                        HalfTriadPolicy::AllowStreamKFixedOrder
                    } else {
                        HalfTriadPolicy::TiledParity
                    });
                }
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
            // Both vendor comparators, then the deterministic routes.
            let t_fast = time_mode(GemmMode::CublasFast, false, false, false, dt);
            let t_pedantic = time_mode(GemmMode::CublasPedantic, false, false, false, dt);
            let t_bi = time_mode(GemmMode::Deterministic, false, false, false, dt);
            eprintln!(
                "[{label} {dt:?}] B={b} T={t}: cuBLAS-Fast {:.3} ms/step | cuBLAS-Pedantic {:.3} ms/step | deterministic scalar {:.3} ms/step | vs Fast {:.2}x | vs Pedantic {:.2}x",
                t_fast * 1e3,
                t_pedantic * 1e3,
                t_bi * 1e3,
                t_bi / t_fast,
                t_bi / t_pedantic
            );
            if dt == WeightDtype::F32 {
                let t_tf32 = time_mode(GemmMode::Deterministic, true, true, false, dt);
                eprintln!(
                    "[{label} {dt:?}] B={b} T={t}: deterministic TF32 {:.3} ms/step | vs Fast {:.2}x | vs Pedantic {:.2}x | vs scalar {:.2}x",
                    t_tf32 * 1e3,
                    t_tf32 / t_fast,
                    t_tf32 / t_pedantic,
                    t_tf32 / t_bi
                );
            } else {
                let t_tc = time_mode(GemmMode::Deterministic, true, false, false, dt);
                eprintln!(
                    "[{label} {dt:?}] B={b} T={t}: deterministic+TC {:.3} ms/step | vs Fast {:.2}x | vs Pedantic {:.2}x | vs scalar {:.2}x",
                    t_tc * 1e3,
                    t_tc / t_fast,
                    t_tc / t_pedantic,
                    t_tc / t_bi
                );
                let t_sk = time_mode(GemmMode::Deterministic, true, false, true, dt);
                eprintln!(
                    "[{label} {dt:?}] B={b} T={t}: deterministic+TC streamk {:.3} ms/step | vs Fast {:.2}x | vs Pedantic {:.2}x | vs TC tiled {:.2}x",
                    t_sk * 1e3,
                    t_sk / t_fast,
                    t_sk / t_pedantic,
                    t_sk / t_tc
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
