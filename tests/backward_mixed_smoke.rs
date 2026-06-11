//! Smoke test for mixed-precision (bf16/f16) M1 backward wiring.
//!
//! Verifies that `gpu_backward_mamba_backbone_mixed` runs end-to-end
//! (forward-mixed + backward-mixed) on a tiny synthetic config without
//! panicking and produces finite (non-NaN, non-inf) master weight
//! gradients. Full numerical parity against the f32 oracle backbone is
//! deferred to Step 6 (M1 mixed training tests — finite-diff + parity
//! + loss curve).

#![cfg(feature = "cuda")]

use mamba_rs::config::MambaConfig;
use mamba_rs::mamba_ssm::gpu::backward_mixed::gpu_backward_mamba_backbone_mixed;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::forward::{GpuMambaDims, GpuRecurrentState};
use mamba_rs::mamba_ssm::gpu::forward_mixed::{
    GpuMambaBackboneMixedActs, GpuMambaMixedTrainScratch, gpu_forward_mamba_backbone_train_mixed,
};
use mamba_rs::mamba_ssm::gpu::weights::GpuMambaGrads;
use mamba_rs::mamba_ssm::gpu::weights_mixed_train::GpuMambaTrainMixedWeights;
use mamba_rs::weights::MambaWeights;

fn tiny_cfg() -> MambaConfig {
    MambaConfig {
        d_model: 32,
        n_layers: 1,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    }
}

fn run_smoke(dtype: WeightDtype) {
    let cfg = tiny_cfg();
    let batch = 1usize;
    let seq_len = 4usize;
    let d_model = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let d_conv = cfg.d_conv;
    let dt_rank = cfg.dt_rank();
    let xdbl_dim = cfg.xdbl_dim();
    let n_layers = cfg.n_layers;

    // CPU weights (identity input_proj — required by mixed forward).
    let mut cpu = MambaWeights::init(&cfg, d_model, 0xBADF00D);
    cpu.input_proj_w.clear();
    cpu.input_proj_b.clear();
    // Recompute a_neg from a_log (init leaves it zero in some paths).
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    // GPU context + weights.
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let weights = GpuMambaTrainMixedWeights::from_cpu(&ctx.stream, &cpu, &cfg, dtype).unwrap();

    let dims = GpuMambaDims {
        batch,
        d_model,
        d_inner: di,
        d_state: ds,
        d_conv,
        dt_rank,
        xdbl_dim,
        seq_len,
        mamba_input_dim: d_model,
        n_layers,
        scan_mode: mamba_rs::config::ScanMode::Auto,
        rms_norm_eps: 1e-5,
    };

    let mut acts = GpuMambaBackboneMixedActs::new(&ctx.stream, &dims, dtype).unwrap();
    let mut scratch = GpuMambaMixedTrainScratch::new(&ctx.stream, &dims, dtype).unwrap();

    // Recurrent state: zeroed conv/ssm, precomputed a_neg_all from CPU weights.
    let mut state = GpuRecurrentState {
        conv_states: GpuBuffer::zeros(&ctx.stream, n_layers * di * d_conv).unwrap(),
        ssm_states: GpuBuffer::zeros(&ctx.stream, n_layers * di * ds).unwrap(),
        a_neg_all: GpuBuffer::zeros(&ctx.stream, n_layers * di * ds).unwrap(),
    };
    let mut a_neg_flat = vec![0.0f32; n_layers * di * ds];
    for (l, lw) in cpu.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg_flat[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }
    state.a_neg_all.upload(&ctx.stream, &a_neg_flat).unwrap();
    ctx.stream.synchronize().unwrap();

    // Deterministic synthetic input (f32 mamba_input).
    let input_vals: Vec<f32> = (0..batch * seq_len * d_model)
        .map(|i| ((i * 131 + 7) & 0xFFFF) as f32 / 65536.0 - 0.5)
        .collect();
    let mut mamba_input = GpuBuffer::zeros(&ctx.stream, batch * seq_len * d_model).unwrap();
    ctx.stream.synchronize().unwrap();
    mamba_input.upload(&ctx.stream, &input_vals).unwrap();
    ctx.stream.synchronize().unwrap();

    // Forward (mixed) — fills acts with saved activations for backward.
    gpu_forward_mamba_backbone_train_mixed(
        &ctx,
        &mut acts,
        &weights,
        &mamba_input,
        &mut state,
        &mut scratch,
    )
    .unwrap();

    // Upstream gradient: random f32 d_temporal.
    let d_temp_vals: Vec<f32> = (0..batch * seq_len * d_model)
        .map(|i| ((i * 17 + 3) & 0xFFFF) as f32 / 65536.0 - 0.5)
        .collect();
    let mut d_temporal = GpuBuffer::zeros(&ctx.stream, batch * seq_len * d_model).unwrap();
    ctx.stream.synchronize().unwrap();
    d_temporal.upload(&ctx.stream, &d_temp_vals).unwrap();
    ctx.stream.synchronize().unwrap();

    // Grads buffer (f32 flat arena).
    let mut d_mamba = GpuMambaGrads::new(&ctx.stream, &cfg, d_model).unwrap();
    d_mamba.zero(&ctx.stream).unwrap();
    ctx.stream.synchronize().unwrap();

    // Backward (mixed). Must not panic.
    gpu_backward_mamba_backbone_mixed(
        &ctx,
        &mut d_temporal,
        &d_mamba,
        &acts,
        &weights.compute,
        &state.a_neg_all,
        &mut scratch,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();

    // Assert all weight grads are finite.
    let mut d_temp_out = vec![0f32; batch * seq_len * d_model];
    d_temporal.download(&ctx.stream, &mut d_temp_out).unwrap();
    assert!(
        d_temp_out.iter().all(|v| v.is_finite()),
        "{dtype:?}: d_temporal contains NaN/inf"
    );

    // Download the flat grads arena and scan for non-finite values.
    let arena_elems = d_mamba.flat.len();
    let mut arena = vec![0f32; arena_elems];
    d_mamba.flat.download(&ctx.stream, &mut arena).unwrap();
    let nonzero = arena.iter().filter(|&&v| v != 0.0).count();
    let nonfinite = arena.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        nonfinite, 0,
        "{dtype:?}: {nonfinite} non-finite values in f32 master grad arena"
    );
    // Sanity: at least some grads are nonzero (backward actually ran).
    assert!(
        nonzero > 0,
        "{dtype:?}: f32 master grad arena is all zeros after backward"
    );
    eprintln!(
        "smoke {dtype:?}: d_temporal finite ✓, grads finite ✓, nonzero={nonzero}/{arena_elems}"
    );
}

#[test]
fn backward_mixed_bf16_smoke() {
    run_smoke(WeightDtype::Bf16);
}

#[test]
fn backward_mixed_f16_smoke() {
    run_smoke(WeightDtype::F16);
}

// NOTE: f32 is intentionally unsupported by the mixed backbone — the pure
// f32 forward/backward path (`gpu_forward_mamba_backbone` +
// `gpu_backward_mamba_backbone`) is what callers use when they don't need
// mixed precision. The mixed forward uses `rmsnorm_fwd_f32in_typed`
// (HalfKernel, bf16/f16 only by construction) so dispatching F32 through
// the mixed path would panic at forward time, not at backward.
