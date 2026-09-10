//! Decode step rate of the M1 backbone at state capacities 64, 128 and
//! 256: the measured cost of the capacity knob. Parity at these
//! capacities is tested in tests/large_d_state.rs.
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

/// Measured cost of the capacity knob: decode steps/second at the
/// baseline d_state=64 vs the register-spilling 128 and 256. Run
/// manually with --ignored --nocapture; the assertion is a sanity
/// factor, not a benchmark claim.
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

// Run every instrument, or only the ones named on the command line:
// `cargo bench --bench <target> --features cuda -- <name> [<name> ...]`.
fn main() {
    let selected: Vec<String> = std::env::args().skip(1).collect();
    let run = |name: &str| selected.is_empty() || selected.iter().any(|s| s == name);
    if run("m1_step_cost_across_state_capacities") {
        m1_step_cost_across_state_capacities();
    }
}
