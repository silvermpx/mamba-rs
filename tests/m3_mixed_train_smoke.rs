//! Step 7 smoke test: `GpuMamba3TrainMixedWeights` + `GpuMamba3BackboneMixedActs`.
//!
//! Validates:
//! 1. Allocation of master (f32) + compute (typed) + typed acts without panic.
//! 2. `sync_master_to_compute` runs both f32-passthrough (D2D) and typed-cast
//!    (f32 → bf16/f16) paths.
//! 3. Typed compute view after sync round-trips back to f32 within bf16/f16
//!    ULP of the master (catches wrong dst ptr / wrong tensor / stale cast).

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::forward_mixed::GpuMamba3BackboneMixedActs;
use mamba_rs::mamba3_siso::gpu::weights_mixed_train::GpuMamba3TrainMixedWeights;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn tiny_m3_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
    }
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    (dot / (na * nb)) as f32
}

fn run_sync(dtype: WeightDtype) {
    let cfg = tiny_m3_cfg();
    let input_dim = cfg.d_model;
    let cpu = Mamba3Weights::init(&cfg, input_dim, 0xBADF00D3);

    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();

    let w =
        GpuMamba3TrainMixedWeights::from_cpu(&ctx.stream, &cpu, &cfg, input_dim, dtype).unwrap();

    // Acts for a tiny shape (B=1, T=4).
    let acts = GpuMamba3BackboneMixedActs::new(&ctx.stream, &cfg, 1, 4, input_dim, dtype).unwrap();
    assert_eq!(acts.layers.len(), cfg.n_layers);
    assert_eq!(acts.dtype, dtype);

    // Sync master → compute.
    w.sync_master_to_compute(&ctx).unwrap();
    ctx.stream.synchronize().unwrap();

    // Spot-check: download `layer[0].in_proj_w` from both master (f32) and
    // compute (typed). They should agree within dtype ULP cosine.
    let mw = &w.master.layers[0];
    let cw = &w.compute.layers[0];
    let n = mw.in_proj_w.len();
    assert_eq!(n, cw.in_proj_w.len_elems());

    let mut master_vals = vec![0f32; n];
    mw.in_proj_w
        .download(&ctx.stream, &mut master_vals)
        .unwrap();

    let mut compute_vals = vec![0f32; n];
    cw.in_proj_w
        .download_to_f32(&ctx.stream, &mut compute_vals)
        .unwrap();
    ctx.stream.synchronize().unwrap();

    let cos = cos_sim(&master_vals, &compute_vals);
    eprintln!("M3 sync {dtype:?}: in_proj_w cos(master, compute_f32_roundtrip)={cos:.6}");
    let tol = match dtype {
        WeightDtype::Bf16 => 0.9995_f32,
        WeightDtype::F16 => 0.99999_f32,
        WeightDtype::F32 => 0.999999_f32,
    };
    assert!(
        cos >= tol,
        "{dtype:?}: master vs compute cos {cos:.6} < {tol}"
    );

    // f32-path check: `norm_weight` is f32-stays-f32; should match bit-
    // exactly after D2D copy.
    let n_norm = mw.norm_weight.len();
    let mut master_norm = vec![0f32; n_norm];
    mw.norm_weight
        .download(&ctx.stream, &mut master_norm)
        .unwrap();
    let mut compute_norm = vec![0f32; n_norm];
    cw.norm_weight
        .download_to_f32(&ctx.stream, &mut compute_norm)
        .unwrap();
    ctx.stream.synchronize().unwrap();
    for (m, c) in master_norm.iter().zip(&compute_norm) {
        assert_eq!(m.to_bits(), c.to_bits(), "norm_weight D2D not bit-exact");
    }
    eprintln!("  norm_weight f32 D2D: bit-exact ✓");
}

#[test]
fn m3_mixed_weights_sync_bf16() {
    run_sync(WeightDtype::Bf16);
}

#[test]
fn m3_mixed_weights_sync_f16() {
    run_sync(WeightDtype::F16);
}

#[test]
fn m3_mixed_weights_sync_f32() {
    run_sync(WeightDtype::F32);
}
