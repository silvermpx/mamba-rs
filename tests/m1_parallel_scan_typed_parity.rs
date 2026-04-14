//! Step 8b — parity test for typed M1 parallel prefix scan forward
//! (`ssm_parallel_scan_fwd_{bf16,f16}`) against the f32 oracle. Also
//! validates the warp-mask fix in `warp_inclusive_scan_ab`'s Step 3 call
//! from `block_inclusive_scan_ab` (was deadlocking on Ada/sm_89 with
//! hardcoded 0xffffffff mask when only NWARPS=4 lanes were active).

#![cfg(feature = "cuda")]

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::backward::gpu_backward_mamba_backbone;
use mamba_rs::mamba_ssm::gpu::backward_mixed::gpu_backward_mamba_backbone_mixed;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::forward::{
    GpuMambaBackboneActs, GpuMambaDims, GpuMambaScratch, GpuRecurrentState,
    gpu_forward_mamba_backbone,
};
use mamba_rs::mamba_ssm::gpu::forward_mixed::{
    GpuMambaBackboneMixedActs, GpuMambaMixedTrainScratch, gpu_forward_mamba_backbone_train_mixed,
};
use mamba_rs::mamba_ssm::gpu::weights::{GpuMambaGrads, GpuMambaTrainWeights};
use mamba_rs::mamba_ssm::gpu::weights_mixed_train::GpuMambaTrainMixedWeights;
use mamba_rs::weights::MambaWeights;

fn tiny_cfg() -> MambaConfig {
    MambaConfig {
        d_model: 32,
        n_layers: 1,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Sequential,
    }
}

fn dims_for(cfg: &MambaConfig, batch: usize, seq_len: usize) -> GpuMambaDims {
    GpuMambaDims {
        batch,
        d_model: cfg.d_model,
        d_inner: cfg.d_inner(),
        d_state: cfg.d_state,
        d_conv: cfg.d_conv,
        dt_rank: cfg.dt_rank(),
        xdbl_dim: cfg.xdbl_dim(),
        seq_len,
        mamba_input_dim: cfg.d_model,
        n_layers: cfg.n_layers,
    }
}

fn build_weights(cfg: &MambaConfig, seed: u64) -> (MambaWeights, MambaWeights) {
    let mut w = MambaWeights::init(cfg, cfg.d_model, seed);
    for lw in w.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    let mut w_f32 = w.clone();
    w_f32.input_proj_w = (0..cfg.d_model * cfg.d_model)
        .map(|i| {
            let r = i / cfg.d_model;
            let c = i % cfg.d_model;
            if r == c { 1.0 } else { 0.0 }
        })
        .collect();
    w_f32.input_proj_b = vec![0.0; cfg.d_model];
    let mut w_mixed = w;
    w_mixed.input_proj_w.clear();
    w_mixed.input_proj_b.clear();
    (w_f32, w_mixed)
}

fn det_rand(n: usize, seed: u32) -> Vec<f32> {
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

fn cos_and_norm(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    let cos = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        1.0
    };
    let ratio = if na > 0.0 {
        ((nb.sqrt()) / (na.sqrt())) as f32
    } else {
        1.0
    };
    (cos, ratio)
}

fn run_f32(
    ctx: &GpuCtx,
    cpu: &MambaWeights,
    cfg: &MambaConfig,
    dims: &GpuMambaDims,
    mamba_input: &[f32],
    d_temporal_init: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let w = GpuMambaTrainWeights::from_cpu(&ctx.stream, cpu).unwrap();
    let mut acts = GpuMambaBackboneActs::new(&ctx.stream, dims).unwrap();
    let mut scratch = GpuMambaScratch::new(&ctx.stream, dims).unwrap();
    let (di, ds, nl, dc) = (dims.d_inner, dims.d_state, dims.n_layers, dims.d_conv);
    let mut state = GpuRecurrentState {
        conv_states: GpuBuffer::zeros(&ctx.stream, nl * di * dc).unwrap(),
        ssm_states: GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap(),
        a_neg_all: GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap(),
    };
    let mut a_neg_flat = vec![0.0f32; nl * di * ds];
    for (l, lw) in cpu.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg_flat[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }
    state.a_neg_all.upload(&ctx.stream, &a_neg_flat).unwrap();
    ctx.stream.synchronize().unwrap();
    let bt = dims.bt();
    let mut input = GpuBuffer::zeros(&ctx.stream, bt * dims.mamba_input_dim).unwrap();
    ctx.stream.synchronize().unwrap();
    input.upload(&ctx.stream, mamba_input).unwrap();
    let mut temporal = GpuBuffer::zeros(&ctx.stream, bt * dims.d_model).unwrap();
    ctx.stream.synchronize().unwrap();
    gpu_forward_mamba_backbone(
        ctx,
        &mut temporal,
        &mut acts,
        &w,
        &input,
        &mut state,
        &mut scratch,
    )
    .unwrap();
    let mut d_temporal = GpuBuffer::zeros(&ctx.stream, bt * dims.d_model).unwrap();
    ctx.stream.synchronize().unwrap();
    d_temporal.upload(&ctx.stream, d_temporal_init).unwrap();
    ctx.stream.synchronize().unwrap();
    let mut grads = GpuMambaGrads::new(&ctx.stream, cfg, dims.mamba_input_dim).unwrap();
    grads.zero(&ctx.stream).unwrap();
    ctx.stream.synchronize().unwrap();
    gpu_backward_mamba_backbone(
        ctx,
        &mut d_temporal,
        &grads,
        &acts,
        &w,
        &state.a_neg_all,
        &mut scratch,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let mut dt_out = vec![0f32; bt * dims.d_model];
    d_temporal.download(&ctx.stream, &mut dt_out).unwrap();
    let mut arena = vec![0f32; grads.flat.len()];
    grads.flat.download(&ctx.stream, &mut arena).unwrap();
    (dt_out, arena)
}

fn run_mixed(
    ctx: &GpuCtx,
    cpu: &MambaWeights,
    cfg: &MambaConfig,
    dims: &GpuMambaDims,
    dtype: WeightDtype,
    mamba_input: &[f32],
    d_temporal_init: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let w = GpuMambaTrainMixedWeights::from_cpu(&ctx.stream, cpu, cfg, dtype).unwrap();
    let mut acts = GpuMambaBackboneMixedActs::new(&ctx.stream, dims, dtype).unwrap();
    let mut scratch = GpuMambaMixedTrainScratch::new(&ctx.stream, dims, dtype).unwrap();
    let (di, ds, nl, dc) = (dims.d_inner, dims.d_state, dims.n_layers, dims.d_conv);
    let mut state = GpuRecurrentState {
        conv_states: GpuBuffer::zeros(&ctx.stream, nl * di * dc).unwrap(),
        ssm_states: GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap(),
        a_neg_all: GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap(),
    };
    let mut a_neg_flat = vec![0.0f32; nl * di * ds];
    for (l, lw) in cpu.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg_flat[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }
    state.a_neg_all.upload(&ctx.stream, &a_neg_flat).unwrap();
    ctx.stream.synchronize().unwrap();
    let bt = dims.bt();
    let mut input = GpuBuffer::zeros(&ctx.stream, bt * dims.mamba_input_dim).unwrap();
    ctx.stream.synchronize().unwrap();
    input.upload(&ctx.stream, mamba_input).unwrap();
    gpu_forward_mamba_backbone_train_mixed(ctx, &mut acts, &w, &input, &mut state, &mut scratch)
        .unwrap();
    let mut d_temporal = GpuBuffer::zeros(&ctx.stream, bt * dims.d_model).unwrap();
    ctx.stream.synchronize().unwrap();
    d_temporal.upload(&ctx.stream, d_temporal_init).unwrap();
    ctx.stream.synchronize().unwrap();
    let mut grads = GpuMambaGrads::new(&ctx.stream, cfg, dims.mamba_input_dim).unwrap();
    grads.zero(&ctx.stream).unwrap();
    ctx.stream.synchronize().unwrap();
    gpu_backward_mamba_backbone_mixed(
        ctx,
        &mut d_temporal,
        &grads,
        &acts,
        &w.compute,
        &state.a_neg_all,
        &mut scratch,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let mut dt_out = vec![0f32; bt * dims.d_model];
    d_temporal.download(&ctx.stream, &mut dt_out).unwrap();
    let mut arena = vec![0f32; grads.flat.len()];
    grads.flat.download(&ctx.stream, &mut arena).unwrap();
    (dt_out, arena)
}

fn check_at_t(dtype: WeightDtype, seq_len: usize) {
    let cfg = tiny_cfg();
    let dims = dims_for(&cfg, 1, seq_len);
    let (w_f32, w_mix) = build_weights(&cfg, 0xDEAD00F1);

    let bt = dims.bt();
    let input = det_rand(bt * dims.mamba_input_dim, 0xA1);
    let d_temporal = det_rand(bt * dims.d_model, 0xA2);

    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();

    let (dt_ref, grads_ref) = run_f32(&ctx, &w_f32, &cfg, &dims, &input, &d_temporal);
    let (dt_typ, grads_typ) = run_mixed(&ctx, &w_mix, &cfg, &dims, dtype, &input, &d_temporal);
    assert_eq!(grads_ref.len(), grads_typ.len());

    let (cos_dt, ratio_dt) = cos_and_norm(&dt_ref, &dt_typ);
    // Skip input_proj_w/b — mixed path uses identity (no grad), f32 path uses
    // eye+zero bias (real grad); arena-wide comparison would be artificially
    // deflated by this structural mismatch, not a kernel bug.
    let input_proj_skip = cfg.d_model * cfg.d_model + cfg.d_model;
    let (cos_g, ratio_g) =
        cos_and_norm(&grads_ref[input_proj_skip..], &grads_typ[input_proj_skip..]);
    eprintln!(
        "{dtype:?} T={seq_len}: d_temporal cos={cos_dt:.6} n={ratio_dt:.4} | \
         grads cos={cos_g:.6} n={ratio_g:.4}"
    );

    let (cos_min, norm_tol) = match dtype {
        WeightDtype::Bf16 => (0.99_f32, 0.10_f32),
        WeightDtype::F16 => (0.995_f32, 0.05_f32),
        WeightDtype::F32 => unreachable!(),
    };
    assert!(
        cos_g >= cos_min,
        "{dtype:?} T={seq_len}: grads cos {cos_g} < {cos_min}"
    );
    assert!((ratio_g - 1.0).abs() <= norm_tol);
    assert!(cos_dt >= cos_min);
    assert!((ratio_dt - 1.0).abs() <= norm_tol);
}

// T=260 forces parallel dispatch (> PARALLEL_SCAN_THRESHOLD=256).
#[test]
fn m1_parallel_scan_typed_parity_bf16_t260() {
    check_at_t(WeightDtype::Bf16, 260);
}

#[test]
fn m1_parallel_scan_typed_parity_f16_t260() {
    check_at_t(WeightDtype::F16, 260);
}

// Longer T amplifies compounding precision drift.
#[test]
fn m1_parallel_scan_typed_parity_bf16_t512() {
    check_at_t(WeightDtype::Bf16, 512);
}

#[test]
fn m1_parallel_scan_typed_parity_f16_t512() {
    check_at_t(WeightDtype::F16, 512);
}
