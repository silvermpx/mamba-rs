//! Large state dimensions on the fused decode path. The step kernels
//! keep the SSM state in per-thread register arrays whose size is a
//! JIT-time capacity knob; these tests prove a d_state past the
//! historical 64 ceiling runs the SAME code path and matches the CPU
//! reference, which has no ceiling and serves as the independent oracle.

#![cfg(feature = "cuda")]

use mamba_rs::MambaBackbone;
use mamba_rs::config::MambaConfig;
use mamba_rs::mamba_ssm::gpu::inference::GpuMambaBackbone;
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

fn check_step_parity(d_state: usize) {
    let cfg = MambaConfig {
        d_model: 64,
        n_layers: 2,
        d_state,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    };
    let input_dim = cfg.d_model;
    let dm = cfg.d_model;

    let cpu_bb = MambaBackbone::init(cfg, input_dim, 42);
    let weights: MambaWeights = cpu_bb.weights().clone();
    let mut state = cpu_bb.alloc_state();
    let mut scratch = cpu_bb.alloc_scratch();

    let mut gpu_bb = GpuMambaBackbone::new(0, &weights, cfg, input_dim, 1).unwrap();

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

#[test]
fn state_capacity_range() {
    use mamba_rs::mamba_ssm::gpu::kernels::state_capacity;
    // 16-granular since P1.1(3): the cap sizes per-thread register
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

/// Measured cost of the capacity knob: decode steps/second at the
/// baseline d_state=64 vs the register-spilling 128 and 256. Run
/// manually with --ignored --nocapture; the assertion is a sanity
/// factor, not a benchmark claim.
#[test]
#[ignore]
fn m1_step_cost_across_state_capacities() {
    let mut rates = Vec::new();
    for ds in [64usize, 128, 256] {
        let cfg = MambaConfig {
            d_model: 256,
            n_layers: 4,
            d_state: ds,
            d_conv: 4,
            expand: 2,
            scan_mode: mamba_rs::config::ScanMode::Sequential,
            rms_norm_eps: 1e-5,
        };
        let input_dim = cfg.d_model;
        let cpu_bb = MambaBackbone::init(cfg, input_dim, 42);
        let weights: MambaWeights = cpu_bb.weights().clone();
        let mut gpu_bb = GpuMambaBackbone::new(0, &weights, cfg, input_dim, 1).unwrap();
        let input = det(input_dim, 0x77);
        let mut out = vec![0.0f32; input_dim];
        for _ in 0..50 {
            gpu_bb.step(&input, &mut out).unwrap();
        }
        let n = 500usize;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            gpu_bb.step(&input, &mut out).unwrap();
        }
        let dt = t0.elapsed().as_secs_f64();
        let rate = n as f64 / dt;
        eprintln!(
            "d_state={ds}: {rate:.0} steps/s ({:.3} ms/step)",
            1e3 * dt / n as f64
        );
        rates.push(rate);
    }
    // Sanity factor: the largest capacity may cost more, but if it is
    // an order of magnitude slower something structural broke (e.g. the
    // arrays spilled somewhere far worse than local memory).
    assert!(
        rates[0] / rates[2] < 10.0,
        "d_state=256 is {}x slower than 64 - structural regression",
        rates[0] / rates[2]
    );
}
