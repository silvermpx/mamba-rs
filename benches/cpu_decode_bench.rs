//! The CPU decode step of the four families side by side: d_model 768,
//! 4 layers, the LM shape (no input projection), microseconds a step
//! (median of runs) with the projections as stored and packed.
//!
//! Run: `cargo bench --bench cpu_decode_bench`

use std::time::Instant;

use mamba_rs::mamba1::MambaConfig;
use mamba_rs::mamba2::Mamba2Config;
use mamba_rs::mamba3_mimo::Mamba3MimoConfig;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::module::{Mamba2Backbone, Mamba3Backbone, Mamba3MimoBackbone, MambaBackbone};

const D_MODEL: usize = 768;
const LAYERS: usize = 4;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0
        })
        .collect()
}

/// The median over `runs` of `steps` decode steps, microseconds a step.
fn time_steps(mut step: impl FnMut(&[f32], &mut [f32]), runs: usize, steps: usize) -> f64 {
    let input = det(D_MODEL, 3);
    let mut out = vec![0f32; D_MODEL];
    for _ in 0..steps {
        step(&input, &mut out);
    }
    let mut times: Vec<f64> = (0..runs)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..steps {
                step(&input, &mut out);
            }
            start.elapsed().as_secs_f64() * 1e6 / steps as f64
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[runs / 2]
}

fn report(name: &str, plain: f64, packed: f64) {
    println!("  {name:<5} stored {plain:>9.1}  packed {packed:>9.1}");
}

fn main() {
    let (runs, steps) = (7, 20);
    println!("--- CPU decode, d_model {D_MODEL}, {LAYERS} layers (us/step) ---");
    {
        let cfg = MambaConfig {
            d_model: D_MODEL,
            n_layers: LAYERS,
            ..MambaConfig::default()
        };
        let mut bb = MambaBackbone::init(cfg, D_MODEL, 1).unwrap();
        let time = |bb: &MambaBackbone| {
            let (mut st, mut sc) = (bb.alloc_state(), bb.alloc_scratch());
            time_steps(|x, y| bb.forward_step(x, y, &mut st, &mut sc), runs, steps)
        };
        let plain = time(&bb);
        bb.pack_for_decode();
        report("M1", plain, time(&bb));
    }
    {
        let cfg = Mamba2Config {
            d_model: D_MODEL,
            n_layers: LAYERS,
            ..Mamba2Config::default()
        };
        let mut bb = Mamba2Backbone::init(cfg, D_MODEL, 2).unwrap();
        let time = |bb: &Mamba2Backbone| {
            let (mut st, mut sc) = (bb.alloc_state(), bb.alloc_scratch());
            time_steps(|x, y| bb.forward_step(x, y, &mut st, &mut sc), runs, steps)
        };
        let plain = time(&bb);
        bb.pack_for_decode();
        report("M2", plain, time(&bb));
    }
    {
        let cfg = Mamba3Config {
            d_model: D_MODEL,
            n_layers: LAYERS,
            ..Mamba3Config::default()
        };
        let mut bb = Mamba3Backbone::init(cfg, D_MODEL, 3).unwrap();
        let time = |bb: &Mamba3Backbone| {
            let (mut st, mut sc) = (bb.alloc_state(), bb.alloc_scratch());
            time_steps(|x, y| bb.forward_step(x, y, &mut st, &mut sc), runs, steps)
        };
        let plain = time(&bb);
        bb.pack_for_decode();
        report("SISO", plain, time(&bb));
    }
    {
        let cfg = Mamba3MimoConfig {
            d_model: D_MODEL,
            n_layers: LAYERS,
            ..Mamba3MimoConfig::default()
        };
        let mut bb = Mamba3MimoBackbone::init(cfg, D_MODEL, 4).unwrap();
        let time = |bb: &Mamba3MimoBackbone| {
            let (mut st, mut sc) = (bb.alloc_state(), bb.alloc_scratch());
            time_steps(|x, y| bb.forward_step(x, y, &mut st, &mut sc), runs, steps)
        };
        let plain = time(&bb);
        bb.pack_for_decode();
        report("MIMO", plain, time(&bb));
    }
}
