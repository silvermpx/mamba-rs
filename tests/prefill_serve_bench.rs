//! The inference serve gate: absolute bit hashes at the production serve
//! shape (single vision-classifier page); the headline ms/page instrument
//! is tools/qualification/prefill_serve_headline.rs. A
//! RELATIVE prefill test (prefill vs training forward) is not enough —
//! a shared-kernel regression moved both sides and stayed green; these
//! hashes are absolute, recorded once per good build.
//!
//!   cargo test --release --features cuda --test prefill_serve_bench -- --ignored --nocapture

#![cfg(feature = "cuda")]

#[path = "common/digest.rs"]
mod digest;
#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/evidence_digest.rs"]
mod evidence_digest;
#[path = "common/hash_outputs.rs"]
mod hash_outputs;
#[path = "common/stamp.rs"]
mod stamp;

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::backward::GpuMambaTargetScratch;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
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

/// Vision-classifier serve geometry (T_TOTAL, INPUT_DIM) — printed in
/// every stamp so a drift is visible at the consumer when the upstream
/// patchifier changes.
const SERVE_T: usize = 4621;
const SERVE_INPUT_DIM: usize = 1024;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

struct Rig {
    device: GpuDevice,
    ctx: GpuCtx,
    weights: GpuMambaWeights,
    scratch: GpuMambaTargetScratch,
    state: GpuInferenceState,
    a_neg: GpuBuffer,
    input: GpuBuffer,
    dims: GpuMambaDims,
    cfg: MambaConfig,
}

fn rig(state_cap: usize) -> Rig {
    let cfg = MambaConfig {
        d_model: 384,
        n_layers: 24,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    };
    let mid = SERVE_INPUT_DIM;
    let mut w = MambaWeights::init(&cfg, mid, 0x5E12E);
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
        seq_len: SERVE_T,
        mamba_input_dim: mid,
        n_layers: nl,
        scan_mode: cfg.scan_mode,
        rms_norm_eps: cfg.rms_norm_eps,
    };
    let device = GpuDevice::new(0).expect("device");
    let ctx = GpuCtx::new_with_state_cap(&device, state_cap).expect("ctx");
    let weights = GpuMambaWeights::from_cpu(&ctx.stream, &w, &cfg).expect("weights");
    let scratch = GpuMambaTargetScratch::new(&ctx.stream, &dims).expect("scratch");
    let state = GpuInferenceState::zeros(&ctx.stream, 1, &cfg).expect("state");
    let mut a_neg_flat = vec![0.0f32; nl * di * ds];
    for (l, lw) in w.layers.iter().enumerate() {
        a_neg_flat[l * di * ds..(l + 1) * di * ds].copy_from_slice(&lw.a_neg);
    }
    let mut a_neg = GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap();
    a_neg.upload(&ctx.stream, &a_neg_flat).unwrap();
    let host_input = det(SERVE_T * mid, 0x51, 0.05);
    let mut input = GpuBuffer::zeros(&ctx.stream, host_input.len()).unwrap();
    input.upload(&ctx.stream, &host_input).unwrap();
    ctx.stream.synchronize().unwrap();
    Rig {
        device,
        ctx,
        weights,
        scratch,
        state,
        a_neg,
        input,
        dims,
        cfg,
    }
}

fn set_tier(ctx: &GpuCtx, tier: &str) {
    match tier {
        "cublas+tf32" => {
            ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
            ctx.set_bi_tensor_cores(false);
        }
        "cublas" => {
            ctx.set_gemm_mode(GemmMode::CublasPedantic).unwrap();
            ctx.set_bi_tensor_cores(false);
        }
        "bi" => {
            ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
            ctx.set_bi_tensor_cores(false);
        }
        "bi+tc" => {
            ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
            ctx.set_bi_tensor_cores(true);
        }
        other => panic!("unknown tier {other}"),
    }
}

/// I-5a: the absolute bit gate — full temporal, last row, pooled sum and
/// the post-prefill inference state, hashed per (tier, state_cap,
/// incoming-state) cell. The non-zero incoming-state cell is the one that
/// can falsify any future conv seeding change.
#[test]
#[ignore = "bit-gate recorder"]
fn prefill_serve_output_hashes() {
    for state_cap in [16usize, 64] {
        for tier in ["cublas+tf32", "cublas", "bi", "bi+tc"] {
            for warm_state in [false, true] {
                let mut r = rig(state_cap);
                set_tier(&r.ctx, tier);
                let dm = r.dims.d_model;
                if warm_state {
                    let conv_len = r.cfg.n_layers * r.cfg.d_inner() * r.cfg.d_conv;
                    let ssm_len = r.cfg.n_layers * r.cfg.d_inner() * r.cfg.d_state;
                    r.state
                        .conv
                        .upload(&r.ctx.stream, &det(conv_len, 21, 0.1))
                        .unwrap();
                    r.state
                        .ssm
                        .upload(&r.ctx.stream, &det(ssm_len, 22, 0.1))
                        .unwrap();
                }
                let mut last = GpuBuffer::zeros(&r.ctx.stream, dm).unwrap();
                let mut full = GpuBuffer::zeros(&r.ctx.stream, SERVE_T * dm).unwrap();
                r.ctx.stream.synchronize().unwrap();
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
                .unwrap();
                let mut pooled = GpuBuffer::zeros(&r.ctx.stream, dm).unwrap();
                let mut r2 = rig(state_cap);
                set_tier(&r2.ctx, tier);
                gpu_forward_inference_prefill_pooled_sum_from_raw(
                    &r2.ctx,
                    &mut pooled,
                    PrefillRawInputs {
                        input_flat: &r2.input,
                        weights: &r2.weights,
                        a_neg_all: &r2.a_neg,
                    },
                    &mut r2.state,
                    &mut r2.scratch,
                )
                .unwrap();
                // pooled was produced on r2's context/stream — hash it there.
                eprintln!(
                    "{}",
                    stamp::bench_stamp(
                        &r.device,
                        &r.ctx,
                        &format!("serve B1 T{SERVE_T} d384 L24 warm_state={warm_state}"),
                        "prefill_from_raw",
                        0
                    )
                );
                hash_outputs::hash_outputs(
                    &r.ctx,
                    &[
                        ("full_temporal", &full, SERVE_T * dm),
                        ("last_temporal", &last, dm),
                        (
                            "state_conv",
                            &r.state.conv,
                            r.cfg.n_layers * r.cfg.d_inner() * r.cfg.d_conv,
                        ),
                        (
                            "state_ssm",
                            &r.state.ssm,
                            r.cfg.n_layers * r.cfg.d_inner() * r.cfg.d_state,
                        ),
                    ],
                );
                hash_outputs::hash_outputs(&r2.ctx, &[("pooled_sum", &pooled, dm)]);
            }
        }
    }
}
