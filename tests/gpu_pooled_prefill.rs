//! 0.5.3 E-surface oracle: the on-device pooled-sum prefill must reproduce
//! the full-temporal download + host column-sum BIT-FOR-BIT (the kernel
//! walks ascending t accumulating pure f32 adds from 0.0 — the same
//! reduction order as a per-column host loop), and the pinned staging
//! buffer must be transfer-transparent (same bytes as a pageable Vec).

#![cfg(feature = "cuda")]

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::backward::GpuMambaTargetScratch;
use mamba_rs::mamba_ssm::gpu::buffers::{GpuBuffer, PinnedHostBuf};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::forward::GpuMambaDims;
use mamba_rs::mamba_ssm::gpu::inference::GpuInferenceState;
use mamba_rs::mamba_ssm::gpu::prefill::{
    PrefillOutputs, PrefillRawInputs, gpu_forward_inference_prefill_from_raw,
    gpu_forward_inference_prefill_pooled_sum_from_raw,
};
use mamba_rs::mamba_ssm::gpu::weights::GpuMambaWeights;
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

struct Rig {
    ctx: GpuCtx,
    weights: GpuMambaWeights,
    scratch: GpuMambaTargetScratch,
    state: GpuInferenceState,
    a_neg: GpuBuffer,
    input: GpuBuffer,
    dims: GpuMambaDims,
}

fn rig(seq_len: usize) -> Rig {
    let cfg = MambaConfig {
        d_model: 48,
        n_layers: 2,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    };
    let mid = 96usize;
    let mut w = MambaWeights::init_with_input(&cfg, mid, 0xBEEF);
    for lw in w.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    let (di, ds, nl) = (cfg.d_inner(), cfg.d_state, cfg.n_layers);
    let dims = GpuMambaDims {
        batch: 1,
        d_model: cfg.d_model,
        d_inner: di,
        d_state: ds,
        d_conv: cfg.d_conv,
        dt_rank: cfg.dt_rank(),
        xdbl_dim: cfg.xdbl_dim(),
        seq_len,
        mamba_input_dim: mid,
        n_layers: nl,
        scan_mode: cfg.scan_mode,
        rms_norm_eps: cfg.rms_norm_eps,
    };
    let device = GpuDevice::new(0).expect("device");
    let ctx = GpuCtx::new(&device).expect("ctx");
    let weights = GpuMambaWeights::from_cpu(&ctx.stream, &w, &cfg).expect("weights");
    let scratch = GpuMambaTargetScratch::new(&ctx.stream, &dims).expect("scratch");
    let state = GpuInferenceState::zeros(&ctx.stream, 1, &cfg).expect("state");
    let mut a_neg_flat = vec![0.0f32; nl * di * ds];
    for (l, lw) in w.layers.iter().enumerate() {
        a_neg_flat[l * di * ds..(l + 1) * di * ds].copy_from_slice(&lw.a_neg);
    }
    let mut a_neg = GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap();
    a_neg.upload(&ctx.stream, &a_neg_flat).unwrap();
    let host_input = det(seq_len * mid, 0x51, 0.05);
    let mut input = GpuBuffer::zeros(&ctx.stream, host_input.len()).unwrap();
    input.upload(&ctx.stream, &host_input).unwrap();
    Rig {
        ctx,
        weights,
        scratch,
        state,
        a_neg,
        input,
        dims,
    }
}

/// pooled_sum == host column-sum of the full temporal, every bit, above
/// and below the parallel-scan threshold.
#[test]
fn pooled_sum_matches_host_colsum_bitwise() {
    for t in [64usize, 333] {
        let mut r = rig(t);
        let dm = r.dims.d_model;
        // Route 1: full temporal -> host colsum (ascending t, from 0.0).
        let mut last = GpuBuffer::zeros(&r.ctx.stream, dm).unwrap();
        let mut full = GpuBuffer::zeros(&r.ctx.stream, t * dm).unwrap();
        r.state.reset(&r.ctx.stream).unwrap();
        gpu_forward_inference_prefill_from_raw(
            &r.ctx,
            PrefillOutputs {
                last_temporal: &mut last,
                full_temporal: Some(&mut full),
            },
            PrefillRawInputs {
                input_flat: &r.input,
                weights: &r.weights,
                a_neg_all: &r.a_neg,
            },
            &mut r.state,
            &mut r.scratch,
        )
        .expect("full prefill");
        let mut temporal = vec![0.0f32; t * dm];
        full.download(&r.ctx.stream, &mut temporal).unwrap();
        let mut host_sum = vec![0.0f32; dm];
        for (j, s) in host_sum.iter_mut().enumerate() {
            for row in 0..t {
                *s += temporal[row * dm + j];
            }
        }
        // Route 2: on-device pooled sum.
        let mut pooled = GpuBuffer::zeros(&r.ctx.stream, dm).unwrap();
        r.state.reset(&r.ctx.stream).unwrap();
        gpu_forward_inference_prefill_pooled_sum_from_raw(
            &r.ctx,
            &mut pooled,
            PrefillRawInputs {
                input_flat: &r.input,
                weights: &r.weights,
                a_neg_all: &r.a_neg,
            },
            &mut r.state,
            &mut r.scratch,
        )
        .expect("pooled prefill");
        let mut device_sum = vec![0.0f32; dm];
        pooled.download(&r.ctx.stream, &mut device_sum).unwrap();
        for (j, (a, b)) in host_sum.iter().zip(&device_sum).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "T={t} dim {j}: host {a} vs device {b}"
            );
        }
    }
}

/// Pinned staging is transfer-transparent: uploading through a
/// PinnedHostBuf and downloading into one moves the exact same bytes as
/// pageable Vec transfers.
#[test]
fn pinned_staging_is_byte_transparent() {
    let device = GpuDevice::new(0).expect("device");
    let ctx = GpuCtx::new(&device).expect("ctx");
    let n = 4621 * 64;
    let host = det(n, 0xA11, 1.0);
    let mut pinned = PinnedHostBuf::zeroed(n).expect("pinned alloc");
    pinned.as_mut_slice().copy_from_slice(&host);
    let mut buf = GpuBuffer::zeros(&ctx.stream, n).unwrap();
    buf.upload(&ctx.stream, pinned.as_slice()).unwrap();
    let mut back_pinned = PinnedHostBuf::zeroed(n).expect("pinned alloc 2");
    buf.download(&ctx.stream, back_pinned.as_mut_slice())
        .unwrap();
    let mut back_pageable = vec![0.0f32; n];
    buf.download(&ctx.stream, &mut back_pageable).unwrap();
    for i in 0..n {
        assert_eq!(host[i].to_bits(), back_pinned.as_slice()[i].to_bits());
        assert_eq!(host[i].to_bits(), back_pageable[i].to_bits());
    }
}
