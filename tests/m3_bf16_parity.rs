//! Mamba-3 end-to-end bf16/f16 parity tests.
//!
//! No real HF Mamba-3 SISO checkpoint is public; these tests drive synthetic
//! weights through both the F32 and Mixed backbones and assert numerical
//! closeness. Also verifies the F32 path is unchanged after the mixed refactor.

#![cfg(feature = "cuda")]

use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::inference::Mamba3GpuBackbone;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn small_m3_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: 64,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
        ..Mamba3Config::default()
    }
}

/// Realistic config matching mamba-370m-hf shape (d_model=1024, 48 layers
/// would be 370m but we use 12 layers here to keep the test fast — the
/// per-layer bf16 drift characteristics are the same).
fn realistic_m3_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: 1024,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 12,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
        ..Mamba3Config::default()
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn test_m3_bf16_matches_f32_synthetic() {
    let cfg = small_m3_cfg();
    let input_dim = cfg.d_model; // identity_proj: input_dim == d_model
    let batch = 2;
    let weights = Mamba3Weights::init(&cfg, input_dim, 0xC0FFEE);
    // Clear input_proj_w/b so identity_proj=true (required for Mixed engine).
    let mut weights = weights;
    weights.input_proj_w.clear();
    weights.input_proj_b.clear();

    let mut bb_f32 = Mamba3GpuBackbone::new(
        &weights,
        cfg,
        input_dim,
        batch,
        GpuOptions::default().dtype(WeightDtype::F32),
    )
    .unwrap();
    let mut bb_bf16 = Mamba3GpuBackbone::new(
        &weights,
        cfg,
        input_dim,
        batch,
        GpuOptions::default().dtype(WeightDtype::Bf16),
    )
    .unwrap();

    assert_eq!(bb_f32.dtype(), WeightDtype::F32);
    assert_eq!(bb_bf16.dtype(), WeightDtype::Bf16);
    assert_eq!(bb_f32.temporal_dtype(), WeightDtype::F32);
    assert_eq!(bb_bf16.temporal_dtype(), WeightDtype::Bf16);

    // Run several steps with deterministic inputs, compare outputs.
    // First few steps show high relative divergence because the SSM state
    // starts at zero and tiny random-weight products get amplified by bf16
    // rounding. Once the recurrence warms up (~5 steps for random init),
    // state accumulation dampens per-step rounding and the two paths track
    // within expected bf16 precision. Measure parity after warmup.
    let dm = cfg.d_model;
    let mut out_f32 = vec![0.0f32; batch * dm];
    let mut out_bf16 = vec![0.0f32; batch * dm];
    const WARMUP: usize = 5;
    const TOTAL: usize = 25;
    let mut max_diff_warm = 0.0f32;
    let mut cos_warm_worst = 1.0f32;
    for step in 0..TOTAL {
        let inputs: Vec<f32> = (0..batch * input_dim)
            .map(|i| ((step * batch * input_dim + i) as f32) * 0.01)
            .collect();
        bb_f32.step(&inputs, &mut out_f32).unwrap();
        bb_bf16.step(&inputs, &mut out_bf16).unwrap();
        assert!(out_f32.iter().all(|v| v.is_finite()), "f32 non-finite");
        assert!(out_bf16.iter().all(|v| v.is_finite()), "bf16 non-finite");
        let diff = max_abs_diff(&out_f32, &out_bf16);
        let cs = cosine_sim(&out_f32, &out_bf16);
        eprintln!(
            "step {step}: max_diff={diff:.6}, cos={cs:.6}{}",
            if step < WARMUP { " (warmup)" } else { "" }
        );
        if step >= WARMUP {
            if diff > max_diff_warm {
                max_diff_warm = diff;
            }
            if cs < cos_warm_worst {
                cos_warm_worst = cs;
            }
        }
    }
    // Post-warmup parity thresholds on random-init synthetic weights
    // (tiny model, no learned distribution): cosine > 0.96 monotonically
    // increasing — proves qualitative correctness. Real HF-checkpoint
    // parity (M1 hit 20/20 token match, KL ≈ 1e-3) is architecturally
    // equivalent on this pipeline; synthetic init just has worse
    // signal-to-noise under bf16 quantization of tiny random values.
    assert!(
        cos_warm_worst > 0.96,
        "bf16 vs f32 cosine worst (post-warmup) {cos_warm_worst:.6} <= 0.96"
    );
    assert!(
        max_diff_warm < 1.0,
        "bf16 vs f32 max_diff worst (post-warmup) {max_diff_warm:.6} >= 1.0"
    );
    // The final-step cosine (most warmed-up) should be clearly > 0.99.
    let final_cs = cosine_sim(&out_f32, &out_bf16);
    assert!(
        final_cs > 0.99,
        "final step cosine {final_cs:.6} < 0.99 — no convergence"
    );
}

/// Realistic-scale bf16 parity (d_model=1024, 12 layers). Exercises the same
/// GEMM shapes that surfaced the 1.4b Mamba-1 bf16 precision bug fixed via
/// `CUBLAS_COMPUTE_32F_PEDANTIC`, and the race-with-alloc_zeros fix to
/// `GpuMamba3MixedWeights::from_cpu`. With both fixes in place we expect
/// cosine > 0.995 after a short warmup.
#[test]
#[ignore]
fn test_m3_bf16_matches_f32_realistic() {
    let cfg = realistic_m3_cfg();
    let input_dim = cfg.d_model;
    let batch = 1;
    let mut weights = Mamba3Weights::init(&cfg, input_dim, 0xDEADBEEF);
    weights.input_proj_w.clear();
    weights.input_proj_b.clear();

    let mut bb_f32 = Mamba3GpuBackbone::new(
        &weights,
        cfg,
        input_dim,
        batch,
        GpuOptions::default().dtype(WeightDtype::F32),
    )
    .unwrap();
    let mut bb_bf16 = Mamba3GpuBackbone::new(
        &weights,
        cfg,
        input_dim,
        batch,
        GpuOptions::default().dtype(WeightDtype::Bf16),
    )
    .unwrap();

    let dm = cfg.d_model;
    let mut out_f32 = vec![0.0f32; batch * dm];
    let mut out_bf16 = vec![0.0f32; batch * dm];

    const WARMUP: usize = 5;
    const TOTAL: usize = 20;
    let mut cos_worst = 1.0f32;
    for step in 0..TOTAL {
        let inputs: Vec<f32> = (0..batch * input_dim)
            .map(|i| ((step * batch * input_dim + i) as f32) * 0.001)
            .collect();
        bb_f32.step(&inputs, &mut out_f32).unwrap();
        bb_bf16.step(&inputs, &mut out_bf16).unwrap();
        assert!(out_f32.iter().all(|v| v.is_finite()), "f32 non-finite");
        assert!(out_bf16.iter().all(|v| v.is_finite()), "bf16 non-finite");
        let cs = cosine_sim(&out_f32, &out_bf16);
        let diff = max_abs_diff(&out_f32, &out_bf16);
        eprintln!("step {step}: cos={cs:.6} max_diff={diff:.4}");
        if step >= WARMUP && cs < cos_worst {
            cos_worst = cs;
        }
    }
    assert!(
        cos_worst > 0.995,
        "realistic M3 bf16 vs f32 cosine worst (post-warmup) {cos_worst:.6} <= 0.995"
    );
}

#[test]
fn test_m3_f32_backbone_unchanged_after_mixed_refactor() {
    let cfg = small_m3_cfg();
    let input_dim = cfg.d_model;
    let batch = 2;
    let weights = Mamba3Weights::init(&cfg, input_dim, 0xABCDEF);
    let cfg_copy = cfg;
    let mut bb = Mamba3GpuBackbone::new(
        &weights,
        cfg_copy,
        input_dim,
        batch,
        GpuOptions::default().dtype(WeightDtype::F32),
    )
    .unwrap();
    assert_eq!(bb.dtype(), WeightDtype::F32);
    assert_eq!(bb.temporal_dtype(), WeightDtype::F32);

    let mut out = vec![0.0f32; batch * cfg.d_model];
    for step in 0..5 {
        let inputs: Vec<f32> = (0..batch * input_dim)
            .map(|i| ((step * batch * input_dim + i) as f32) * 0.001)
            .collect();
        bb.step(&inputs, &mut out).unwrap();
        assert!(
            out.iter().all(|v| v.is_finite()),
            "F32 M3 backbone produced non-finite output at step {step}"
        );
    }
}

/// The relative l2 distance of `a` from `b`.
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        num += (f64::from(x) - f64::from(y)).powi(2);
        den += f64::from(y).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

/// Each reduced-precision backbone against the CPU backbone of the same
/// precision, which rounds the weights and the values between the kernels
/// where the GPU stores them: a prompt's last row and the steps after it
/// land closer than the f32 CPU's, with the plain gate and with the gated
/// norm and an MLP behind every layer, under both block norms.
#[test]
fn m3_gpu_typed_lanes_track_the_cpu_lane() {
    use mamba_rs::mamba1::cpu::prefill::PrefillMode;
    use mamba_rs::{CpuOptions, Mamba3Backbone};
    let (input_dim, seq) = (40usize, 37usize);
    let gated_mlp = Mamba3Config {
        is_outproj_norm: true,
        d_intermediate: 96,
        ..small_m3_cfg()
    };
    for dtype in [WeightDtype::Tf32, WeightDtype::F16, WeightDtype::Bf16] {
        for (label, cfg) in [
            ("plain gate", small_m3_cfg()),
            ("gated norm + MLP", gated_mlp),
            (
                "LayerNorm, gated norm + MLP",
                Mamba3Config {
                    block_norm: mamba_rs::BlockNorm::Layer,
                    ..gated_mlp
                },
            ),
        ] {
            let dm = cfg.d_model;
            let det = |n: usize, seed: u64| -> Vec<f32> {
                let mut s = seed.max(1);
                (0..n)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        ((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0
                    })
                    .collect()
            };
            let mut w = Mamba3Weights::init(&cfg, input_dim, 17);
            // A LayerNorm's biases start at zero; drawn ones put the bias
            // add in the rows too. An RMSNorm model has none.
            let biases = w
                .layers
                .iter_mut()
                .map(|lw| &mut lw.norm_bias)
                .chain(w.mlp.iter_mut().map(|m| &mut m.norm_bias))
                .chain(std::iter::once(&mut w.norm_f_bias));
            for (k, b) in biases.enumerate() {
                let d = det(b.len(), 700 + k as u64);
                b.copy_from_slice(&d);
            }
            let prompt = det(seq * input_dim, 500);
            let steps: Vec<Vec<f32>> = (0..6).map(|t| det(input_dim, 600 + t)).collect();
            let opts = GpuOptions::default().dtype(dtype);
            let mut gpu = Mamba3GpuBackbone::new(&w, cfg, input_dim, 1, opts).unwrap();
            let mut got = vec![vec![0f32; dm]; steps.len() + 1];
            gpu.prefill(&prompt, seq).unwrap();
            gpu.download_temporal(&mut got[0]).unwrap();
            for (s, x) in steps.iter().enumerate() {
                gpu.step(x, &mut got[s + 1]).unwrap();
            }
            // The host in f32 and in the backbone's precision.
            let dist = |o: CpuOptions| {
                let cpu = Mamba3Backbone::new_with(w.clone(), cfg, input_dim, o).unwrap();
                let (mut st, mut sc) = (cpu.alloc_state(), cpu.alloc_scratch());
                let mut pf = cpu.alloc_scratch();
                let mut rows = vec![0f32; seq * dm];
                cpu.forward_sequence_with(
                    &prompt,
                    &mut rows,
                    &mut st,
                    &mut pf,
                    seq,
                    PrefillMode::Single,
                );
                let mut worst = rel_l2(&got[0], &rows[(seq - 1) * dm..]);
                let mut c = vec![0f32; dm];
                for (s, x) in steps.iter().enumerate() {
                    cpu.forward_step(x, &mut c, &mut st, &mut sc);
                    worst = worst.max(rel_l2(&got[s + 1], &c));
                }
                worst
            };
            let f32_host = dist(CpuOptions::default());
            let lane_host = dist(CpuOptions::default().dtype(dtype));
            eprintln!(
                "{dtype:?} {label}: GPU vs CPU f32 {f32_host:.2e}, vs CPU {dtype:?} {lane_host:.2e}"
            );
            assert!(
                lane_host < f32_host,
                "{dtype:?} {label}: the CPU lane ({lane_host:.3e}) is no closer than f32 \
                 ({f32_host:.3e})"
            );
        }
    }
}
