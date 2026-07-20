//! GPU inference prefill vs GPU TRAINING forward — the serving question.
//!
//! The production classifier checkpoints are TRAINED on the GPU forward
//! (`gpu_forward_mamba_backbone`); a GPU serving lane scores through the
//! nosave inference prefill (`gpu_forward_inference_prefill`). This suite
//! pins the claim that the two produce the SAME BITS on the same device:
//! the deterministic kernel set (fixed-order reductions, one arithmetic
//! chain per element) must make save vs nosave a storage difference only.
//!
//! Coverage:
//! - sequential-scan shape (small T) — bitwise, full temporal;
//! - parallel-scan shape (T > threshold, Auto mode both sides) — bitwise;
//! - `_full` entry fed the training forward's OWN input_proj output — pins
//!   the shared SSM/norm chain through the official all-T surface;
//! - `_from_raw` entry fed the RAW input — end-to-end bits INCLUDING the
//!   internal input projection (the training-forward SGEMM call).

#![cfg(feature = "cuda")]

use mamba_rs::config::MambaConfig;
use mamba_rs::mamba_ssm::gpu::backward::GpuMambaTargetScratch;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::forward::{
    GpuMambaBackboneActs, GpuMambaDims, GpuMambaScratch, GpuRecurrentState,
    gpu_forward_mamba_backbone,
};
use mamba_rs::mamba_ssm::gpu::inference::GpuInferenceState;
use mamba_rs::mamba_ssm::gpu::prefill::{
    PrefillInputs, PrefillOutputs, PrefillRawInputs, gpu_forward_inference_prefill_from_raw,
    gpu_forward_inference_prefill_full,
};
use mamba_rs::mamba_ssm::gpu::weights::{GpuMambaTrainWeights, GpuMambaWeights};
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

fn assert_bits(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length");
    let mut worst = 0.0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            worst = worst.max((x - y).abs());
            assert_eq!(x.to_bits(), y.to_bits(), "{label}[{i}]: {x} vs {y}");
        }
    }
    let _ = worst;
}

fn run_case(cfg: &MambaConfig, input_dim: usize, seq_len: usize, seed: u64, label: &str) {
    let mut w = MambaWeights::init(cfg, input_dim, seed);
    for lw in w.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    let batch = 1usize;
    let n_temporal = batch * seq_len * cfg.d_model;
    let di = cfg.d_inner();
    let (ds, dc, nl) = (cfg.d_state, cfg.d_conv, cfg.n_layers);
    let input = det(batch * seq_len * input_dim, 0xA5A5, 0.05);
    let mut a_neg_flat = vec![0.0f32; nl * di * ds];
    for (l, lw) in w.layers.iter().enumerate() {
        a_neg_flat[l * di * ds..(l + 1) * di * ds].copy_from_slice(&lw.a_neg);
    }

    let device = GpuDevice::new(0).expect("gpu device");
    let ctx = GpuCtx::new(&device).expect("gpu ctx");
    let gpu_dims = GpuMambaDims {
        batch,
        d_model: cfg.d_model,
        d_inner: di,
        d_state: ds,
        d_conv: dc,
        dt_rank: cfg.dt_rank(),
        xdbl_dim: cfg.xdbl_dim(),
        seq_len,
        mamba_input_dim: input_dim,
        n_layers: nl,
        scan_mode: cfg.scan_mode,
        rms_norm_eps: cfg.rms_norm_eps,
    };

    // ---- GPU TRAINING forward (the path the checkpoint was trained on) ----
    let gpu_tw = GpuMambaTrainWeights::from_cpu(&ctx.stream, &w).expect("train weights upload");
    let mut acts = GpuMambaBackboneActs::new(&ctx.stream, &gpu_dims).expect("acts");
    let mut tr_scratch = GpuMambaScratch::new(&ctx.stream, &gpu_dims).expect("train scratch");
    let mut tr_state = GpuRecurrentState {
        conv_states: GpuBuffer::zeros(&ctx.stream, nl * di * dc).unwrap(),
        ssm_states: GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap(),
        a_neg_all: GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap(),
    };
    tr_state.a_neg_all.upload(&ctx.stream, &a_neg_flat).unwrap();
    let mut train_temporal = GpuBuffer::zeros(&ctx.stream, n_temporal).unwrap();
    let mut gpu_input = GpuBuffer::zeros(&ctx.stream, input.len()).unwrap();
    gpu_input.upload(&ctx.stream, &input).unwrap();
    gpu_forward_mamba_backbone(
        &ctx,
        &mut train_temporal,
        &mut acts,
        &gpu_tw,
        &gpu_input,
        &mut tr_state,
        &mut tr_scratch,
    )
    .expect("training forward");
    let mut train_out = vec![0.0f32; n_temporal];
    train_temporal
        .download(&ctx.stream, &mut train_out)
        .expect("train temporal");
    let mut ip_out = vec![0.0f32; n_temporal];
    acts.input_proj_outputs
        .download(&ctx.stream, &mut ip_out)
        .expect("ip_out download");

    // ---- GPU INFERENCE prefill, official API ----
    let inf_w = GpuMambaWeights::from_cpu(&ctx.stream, &w, cfg).expect("inference weights");
    let mut inf_state = GpuInferenceState::zeros(&ctx.stream, batch, cfg).expect("inference state");
    let mut inf_scratch = GpuMambaTargetScratch::new(&ctx.stream, &gpu_dims).expect("inf scratch");
    let mut a_neg_buf = GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap();
    a_neg_buf.upload(&ctx.stream, &a_neg_flat).unwrap();
    let mut ip_buf = GpuBuffer::zeros(&ctx.stream, ip_out.len()).unwrap();
    ip_buf.upload(&ctx.stream, &ip_out).unwrap();
    let mut last_temporal = GpuBuffer::zeros(&ctx.stream, batch * cfg.d_model).unwrap();
    let mut full_temporal = GpuBuffer::zeros(&ctx.stream, n_temporal).unwrap();
    let tail = &train_out[(seq_len - 1) * cfg.d_model..seq_len * cfg.d_model];

    // Case A: `_full` fed the training forward's OWN projected input — pins
    // the shared SSM/norm chain through the official all-T out.
    gpu_forward_inference_prefill_full(
        &ctx,
        PrefillOutputs {
            last_temporal: &mut last_temporal,
            full_temporal: Some(&mut full_temporal),
        },
        PrefillInputs {
            ip_out_flat: &ip_buf,
            weights: &inf_w,
            a_neg_all: &a_neg_buf,
        },
        &mut inf_state,
        &mut inf_scratch,
    )
    .expect("inference prefill full");
    let mut inf_out = vec![0.0f32; n_temporal];
    full_temporal
        .download(&ctx.stream, &mut inf_out)
        .expect("full_temporal download");
    assert_bits(label, &inf_out, &train_out);
    let mut last = vec![0.0f32; batch * cfg.d_model];
    last_temporal
        .download(&ctx.stream, &mut last)
        .expect("last temporal");
    assert_bits(&format!("{label}/last-gather"), &last, tail);

    // Case B: `_from_raw` fed the RAW input — the input projection runs
    // inside (training-forward SGEMM), so this pins the END-TO-END bits.
    inf_state.reset(&ctx.stream).expect("state reset");
    let mut raw_buf = GpuBuffer::zeros(&ctx.stream, input.len()).unwrap();
    raw_buf.upload(&ctx.stream, &input).unwrap();
    gpu_forward_inference_prefill_from_raw(
        &ctx,
        PrefillOutputs {
            last_temporal: &mut last_temporal,
            full_temporal: Some(&mut full_temporal),
        },
        PrefillRawInputs {
            input_flat: &raw_buf,
            weights: &inf_w,
            a_neg_all: &a_neg_buf,
        },
        &mut inf_state,
        &mut inf_scratch,
    )
    .expect("inference prefill from_raw");
    full_temporal
        .download(&ctx.stream, &mut inf_out)
        .expect("full_temporal download (from_raw)");
    assert_bits(&format!("{label}/from-raw"), &inf_out, &train_out);
    last_temporal
        .download(&ctx.stream, &mut last)
        .expect("last temporal (from_raw)");
    assert_bits(&format!("{label}/from-raw/last-gather"), &last, tail);
}

/// Sequential-scan shape: the small-T route.
#[test]
fn gpu_inference_prefill_matches_gpu_training_forward_sequential() {
    let cfg = MambaConfig {
        d_model: 32,
        n_layers: 2,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    };
    run_case(&cfg, 20, 8, 0xC0FFEE, "seq-scan");
}

/// Parallel-scan shape: T beyond the Auto threshold — the classifier regime
/// (T=4621 in production rides this kernel).
#[test]
fn gpu_inference_prefill_matches_gpu_training_forward_parallel_scan() {
    let cfg = MambaConfig {
        d_model: 32,
        n_layers: 2,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Auto,
        rms_norm_eps: 1e-5,
    };
    let t = mamba_rs::mamba_ssm::gpu::forward::PARALLEL_SCAN_THRESHOLD + 77;
    run_case(&cfg, 20, t, 0xBEEF, "parallel-scan");
}
