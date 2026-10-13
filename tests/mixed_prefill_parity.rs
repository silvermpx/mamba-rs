//! Numeric oracle for the M-A rewire (0.5.2): the mixed (bf16/f16)
//! inference prefill's PARALLEL scan route vs the sequential typed burnin
//! it used to run unconditionally.
//!
//! The two kernels are different reduction structures over the same
//! algebra, so the contract is tolerance parity (the same law as every
//! sequential-vs-parallel pair in the crate), plus exact agreement of the
//! argmax-relevant magnitudes: outputs and the carried SSM state. The
//! all-T entry's rows end in the plain entry's last rows, bit for bit, and
//! track the f32 prefill's rows.

#![cfg(feature = "cuda")]

use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba1::MambaWeights;
use mamba_rs::mamba1::gpu::backward::{GpuMambaTargetMixedScratch, GpuMambaTargetScratch};
use mamba_rs::mamba1::gpu::forward::GpuMambaDims;
use mamba_rs::mamba1::gpu::inference::MambaGpuState;
use mamba_rs::mamba1::gpu::prefill::{
    PrefillInputs, PrefillMixedOutputs, PrefillOutputs, gpu_forward_inference_prefill_full,
    gpu_forward_inference_prefill_mixed, gpu_forward_inference_prefill_mixed_full,
};
use mamba_rs::mamba1::gpu::weights::{GpuMambaMixedWeights, GpuMambaWeights};
use mamba_rs::{config::ScanMode, mamba1::MambaConfig};

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

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-30)) as f32
}

fn run_route(mode: ScanMode, dtype: WeightDtype, seq_len: usize) -> (Vec<f32>, Vec<f32>) {
    let cfg = MambaConfig {
        d_model: 32,
        n_layers: 2,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: mode,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
        dt_rank: None,
        proj_bias: false,
        conv_bias: true,
    };
    let mut w = MambaWeights::init(&cfg, cfg.d_model, 0xC0FFEE);
    for lw in w.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    let batch = 1usize;
    let (di, ds, dc, nl) = (cfg.d_inner(), cfg.d_state, cfg.d_conv, cfg.n_layers);
    let dims = GpuMambaDims {
        batch,
        d_model: cfg.d_model,
        d_inner: di,
        d_state: ds,
        d_conv: dc,
        dt_rank: cfg.dt_rank(),
        xdbl_dim: cfg.xdbl_dim(),
        seq_len,
        mamba_input_dim: cfg.d_model,
        n_layers: nl,
        scan_mode: cfg.scan_mode,
        rms_norm_eps: cfg.rms_norm_eps,
        bcdt_rms_eps: None,
    };
    let device = GpuDevice::new(0).expect("device");
    let ctx = GpuCtx::new(&device).expect("ctx");
    let weights = GpuMambaMixedWeights::from_cpu(&ctx.stream, &w, &cfg, dtype).expect("weights");
    let mut state = MambaGpuState::zeros(&ctx.stream, batch, &cfg).expect("state");
    let mut scratch = GpuMambaTargetMixedScratch::new(&ctx.stream, &dims, dtype).expect("scratch");
    let mut a_neg_flat = vec![0.0f32; nl * di * ds];
    for (l, lw) in w.layers.iter().enumerate() {
        a_neg_flat[l * di * ds..(l + 1) * di * ds].copy_from_slice(&lw.a_neg);
    }
    let mut a_neg = GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap();
    a_neg.upload(&ctx.stream, &a_neg_flat).unwrap();
    let ip = det(batch * seq_len * cfg.d_model, 0xA5, 0.05);
    let mut ip_buf = GpuBuffer::zeros(&ctx.stream, ip.len()).unwrap();
    ip_buf.upload(&ctx.stream, &ip).unwrap();
    let target = DtypedBuf::zeros(&ctx.stream, batch * cfg.d_model, dtype).expect("target");

    gpu_forward_inference_prefill_mixed(
        &ctx,
        &target,
        PrefillInputs {
            ip_out_flat: &ip_buf,
            weights: &weights,
            a_neg_all: &a_neg,
        },
        &mut state,
        &mut scratch,
    )
    .expect("mixed prefill");

    let mut last = vec![0.0f32; batch * cfg.d_model];
    target
        .download_f32(&ctx.stream, &mut last)
        .expect("target download");
    let mut ssm = vec![0.0f32; nl * batch * di * ds];
    state.ssm.download(&ctx.stream, &mut ssm).expect("ssm");
    (last, ssm)
}

/// The rewired parallel route must agree with the sequential route it
/// replaced — outputs and carried state, both dtypes, T above threshold.
#[test]
fn mixed_prefill_parallel_route_matches_sequential() {
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let t = 333usize; // > threshold: Auto -> parallel typed nosave
        let (last_seq, ssm_seq) = run_route(ScanMode::Sequential, dtype, t);
        let (last_par, ssm_par) = run_route(ScanMode::Auto, dtype, t);
        let c_last = cos_sim(&last_seq, &last_par);
        let c_ssm = cos_sim(&ssm_seq, &ssm_par);
        eprintln!("{dtype:?}: last cos {c_last:.6}  ssm cos {c_ssm:.6}");
        assert!(
            c_last >= 0.999,
            "{dtype:?}: last-temporal diverged: {c_last}"
        );
        assert!(
            c_ssm >= 0.999,
            "{dtype:?}: carried SSM state diverged: {c_ssm}"
        );
    }
}

/// Below the threshold both modes take the sequential kernel — sanity that
/// the dispatch itself does not change short-prompt behavior.
#[test]
fn mixed_prefill_short_prompt_unchanged() {
    let t = 64usize;
    let (last_a, ssm_a) = run_route(ScanMode::Sequential, WeightDtype::Bf16, t);
    let (last_b, ssm_b) = run_route(ScanMode::Auto, WeightDtype::Bf16, t);
    for (i, (x, y)) in last_a.iter().zip(&last_b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "last[{i}] differs below threshold"
        );
    }
    for (i, (x, y)) in ssm_a.iter().zip(&ssm_b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "ssm[{i}] differs below threshold");
    }
}

/// The all-T bf16/f16 prefill over two sequences: its last rows are the
/// plain entry's output and each sequence's last full row, bit for bit,
/// and its rows track the f32 prefill's rows of the same model.
#[test]
fn mixed_prefill_full_rows_end_in_the_last_rows() {
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for t in [64usize, 333] {
            let cfg = MambaConfig {
                d_model: 32,
                n_layers: 2,
                d_state: 16,
                d_conv: 4,
                expand: 2,
                scan_mode: ScanMode::Auto,
                rms_norm_eps: 1e-5,
                bcdt_rms_eps: None,
                dt_rank: None,
                proj_bias: false,
                conv_bias: true,
            };
            let mut w = MambaWeights::init(&cfg, cfg.d_model, 0xC0FFEE);
            for lw in w.layers.iter_mut() {
                lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
            }
            let (batch, dm) = (2usize, cfg.d_model);
            let (di, ds, nl) = (cfg.d_inner(), cfg.d_state, cfg.n_layers);
            let dims = GpuMambaDims {
                batch,
                d_model: dm,
                d_inner: di,
                d_state: ds,
                d_conv: cfg.d_conv,
                dt_rank: cfg.dt_rank(),
                xdbl_dim: cfg.xdbl_dim(),
                seq_len: t,
                mamba_input_dim: dm,
                n_layers: nl,
                scan_mode: cfg.scan_mode,
                rms_norm_eps: cfg.rms_norm_eps,
                bcdt_rms_eps: None,
            };
            let device = GpuDevice::new(0).expect("device");
            let ctx = GpuCtx::new(&device).expect("ctx");
            let s = &ctx.stream;
            let half = GpuMambaMixedWeights::from_cpu(s, &w, &cfg, dtype).expect("weights");
            let mut state = MambaGpuState::zeros(s, batch, &cfg).expect("state");
            let mut scratch = GpuMambaTargetMixedScratch::new(s, &dims, dtype).expect("scratch");
            let mut a_neg_flat = vec![0.0f32; nl * di * ds];
            for (l, lw) in w.layers.iter().enumerate() {
                a_neg_flat[l * di * ds..(l + 1) * di * ds].copy_from_slice(&lw.a_neg);
            }
            let mut a_neg = GpuBuffer::zeros(s, nl * di * ds).unwrap();
            a_neg.upload(s, &a_neg_flat).unwrap();
            let ip = det(batch * t * dm, 0xA5, 0.05);
            let mut ip_buf = GpuBuffer::zeros(s, ip.len()).unwrap();
            ip_buf.upload(s, &ip).unwrap();
            let download = |buf: &DtypedBuf, n: usize| -> Vec<f32> {
                let mut v = vec![0.0f32; n];
                buf.download_f32(s, &mut v).expect("download");
                v
            };
            let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };

            let plain = DtypedBuf::zeros(s, batch * dm, dtype).unwrap();
            gpu_forward_inference_prefill_mixed(
                &ctx,
                &plain,
                PrefillInputs {
                    ip_out_flat: &ip_buf,
                    weights: &half,
                    a_neg_all: &a_neg,
                },
                &mut state,
                &mut scratch,
            )
            .expect("plain prefill");
            let plain = download(&plain, batch * dm);

            state.reset(s).unwrap();
            let last = DtypedBuf::zeros(s, batch * dm, dtype).unwrap();
            let full = DtypedBuf::zeros(s, batch * t * dm, dtype).unwrap();
            gpu_forward_inference_prefill_mixed_full(
                &ctx,
                PrefillMixedOutputs {
                    last_temporal: &last,
                    full_temporal: Some(&full),
                },
                PrefillInputs {
                    ip_out_flat: &ip_buf,
                    weights: &half,
                    a_neg_all: &a_neg,
                },
                &mut state,
                &mut scratch,
            )
            .expect("all-T prefill");
            let (last, full) = (download(&last, batch * dm), download(&full, batch * t * dm));
            assert!(
                bits(&last) == bits(&plain),
                "{dtype:?} T={t}: the all-T entry moved the last rows"
            );
            for b in 0..batch {
                let row = &full[(b * t + t - 1) * dm..(b * t + t) * dm];
                assert!(
                    bits(row) == bits(&last[b * dm..(b + 1) * dm]),
                    "{dtype:?} T={t}: sequence {b}'s last full row is not its last row"
                );
            }

            let f32_weights = GpuMambaWeights::from_cpu(s, &w, &cfg).expect("f32 weights");
            let mut f32_scratch = GpuMambaTargetScratch::new(s, &dims).expect("f32 scratch");
            let mut f32_last = GpuBuffer::zeros(s, batch * dm).unwrap();
            let mut f32_full = GpuBuffer::zeros(s, batch * t * dm).unwrap();
            state.reset(s).unwrap();
            gpu_forward_inference_prefill_full(
                &ctx,
                PrefillOutputs {
                    last_temporal: &mut f32_last,
                    full_temporal: Some(&mut f32_full),
                },
                PrefillInputs {
                    ip_out_flat: &ip_buf,
                    weights: &f32_weights,
                    a_neg_all: &a_neg,
                },
                &mut state,
                &mut f32_scratch,
            )
            .expect("f32 prefill");
            let mut f32_rows = vec![0.0f32; batch * t * dm];
            f32_full.download(s, &mut f32_rows).unwrap();
            let c = cos_sim(&full, &f32_rows);
            eprintln!("{dtype:?} T={t}: all-T rows vs the f32 prefill cos {c:.6}");
            assert!(
                c >= 0.999,
                "{dtype:?} T={t}: all-T rows diverged from f32: {c}"
            );
        }
    }
}
