//! Large state dimensions on the fused decode path. The step kernels
//! keep the SSM state in per-thread register arrays whose size is a
//! JIT-time capacity knob; these tests prove a d_state past the
//! historical 64 ceiling runs the SAME code path and matches the CPU
//! reference, which has no ceiling and serves as the independent oracle.

#![cfg(feature = "cuda")]

use mamba_rs::MambaBackbone;
use mamba_rs::gpu::GpuOptions;
use mamba_rs::mamba1::MambaConfig;
use mamba_rs::mamba1::MambaWeights;
use mamba_rs::mamba1::gpu::inference::MambaGpuBackbone;

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

fn check_step_parity(d_state: usize) {
    let cfg = MambaConfig {
        d_model: 64,
        n_layers: 2,
        d_state,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
        dt_rank: None,
        proj_bias: false,
        conv_bias: true,
        d_intermediate: 0,
    };
    let input_dim = cfg.d_model;
    let dm = cfg.d_model;

    let cpu_bb = MambaBackbone::init(cfg, input_dim, 42);
    let weights: MambaWeights = cpu_bb.weights().clone();
    let mut state = cpu_bb.alloc_state();
    let mut scratch = cpu_bb.alloc_scratch();

    let mut gpu_bb =
        MambaGpuBackbone::new(&weights, cfg, input_dim, 1, GpuOptions::default()).unwrap();

    let steps = 12usize;
    let mut cpu_out = vec![0.0f32; dm];
    let mut gpu_out = vec![0.0f32; dm];
    for t in 0..steps {
        let input = det(input_dim, 0x51 + t as u32);
        cpu_bb.forward_step(&input, &mut cpu_out, &mut state, &mut scratch);
        gpu_bb.step(&input, &mut gpu_out).unwrap();
        let (mut dot, mut na, mut nb, mut num, mut den) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&x, &y) in cpu_out.iter().zip(gpu_out.iter()) {
            dot += (x as f64) * (y as f64);
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
            num += ((x - y) as f64) * ((x - y) as f64);
            den += (x as f64) * (x as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
        let rel = (num / den.max(1e-30)).sqrt();
        // Loose enough for the engine's TF32 GEMM tier accumulating over
        // the recurrent steps; tight enough to catch the capacity-guard
        // failure mode (a kernel returning without writing leaves zeros,
        // which collapses the cosine).
        assert!(
            cos > 0.999 && rel < 2e-2,
            "d_state={d_state} step {t}: cos={cos:.6} rel_l2={rel:.3e} \
             (GPU fused step diverged from the CPU reference)"
        );
    }
}

#[test]
fn m1_gpu_step_matches_cpu_at_d_state_128() {
    check_step_parity(128);
}

#[test]
fn m1_gpu_step_matches_cpu_at_d_state_256() {
    check_step_parity(256);
}

/// An explicit Sequential at 128 states runs the sequential scan and trains
/// like the parallel scan: the forward rows and every gradient agree to
/// rounding, the same recurrence summed in another order. Before, the mode
/// was silently replaced by Parallel above 64 states.
#[test]
fn m1_explicit_sequential_trains_at_d_state_128() {
    use mamba_rs::config::ScanMode;
    use mamba_rs::{BackwardOpts, MambaGpuTrainer, TrainSessionCfg};
    let (batch, seq_len, dm) = (2, 40, 64);
    let run = |scan_mode: ScanMode| -> (Vec<f32>, Vec<f32>) {
        let cfg = MambaConfig {
            d_model: dm,
            n_layers: 2,
            d_state: 128,
            d_conv: 4,
            expand: 2,
            scan_mode,
            rms_norm_eps: 1e-5,
            bcdt_rms_eps: None,
            dt_rank: None,
            proj_bias: false,
            conv_bias: true,
            d_intermediate: 0,
        };
        let w = MambaWeights::init(&cfg, dm, 7);
        let session = TrainSessionCfg {
            input_dim: dm,
            batch,
            seq_len,
            lr: 1e-3,
            weight_decay: 0.0,
        };
        let mut t = MambaGpuTrainer::new(&w, cfg, session, GpuOptions::default()).unwrap();
        let mut rows = vec![0.0f32; batch * seq_len * dm];
        t.forward(&det(batch * seq_len * dm, 0x71), &mut rows)
            .unwrap();
        let d = det(batch * seq_len * dm, 0x72);
        t.backward_step(&d, BackwardOpts::default()).unwrap();
        let stream = t.ctx().stream.clone();
        (rows, t.grad_arena().to_cpu(&stream).unwrap())
    };
    let (seq_rows, seq_grads) = run(ScanMode::Sequential);
    let (par_rows, par_grads) = run(ScanMode::Parallel);
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
        let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
        (num / den.max(1e-30)).sqrt()
    };
    let (r, g) = (rel(&seq_rows, &par_rows), rel(&seq_grads, &par_grads));
    eprintln!("sequential vs parallel at 128 states: rows {r:.2e}, grads {g:.2e}");
    assert!(
        seq_rows != par_rows || seq_grads != par_grads,
        "the explicit Sequential gave the parallel scan's bits: it was not honoured"
    );
    assert!(r < 1e-4, "forward rows differ by {r}");
    assert!(g < 1e-3, "gradients differ by {g}");
}

#[test]
fn state_capacity_range() {
    use mamba_rs::gpu::kernels::state_capacity;
    // 16-granular: the cap sizes per-thread register
    // arrays; a 64-floor at small d_state was 4x local-memory waste.
    assert_eq!(state_capacity(8).unwrap(), 16);
    assert_eq!(state_capacity(16).unwrap(), 16);
    assert_eq!(state_capacity(17).unwrap(), 32);
    assert_eq!(state_capacity(64).unwrap(), 64);
    assert_eq!(state_capacity(65).unwrap(), 80);
    assert_eq!(state_capacity(128).unwrap(), 128);
    assert_eq!(state_capacity(200).unwrap(), 208);
    assert_eq!(state_capacity(256).unwrap(), 256);
    assert!(state_capacity(0).is_err());
    assert!(state_capacity(257).is_err());
}
