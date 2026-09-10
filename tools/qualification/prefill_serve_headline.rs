//! Headline ms/page of the inference prefill at the production serve shape
//! (one vision-classifier page). Manual instrument: build with
//! `--features cuda,qualification` and run the ignored test on a quiet board.
//! The absolute output hashes of the same shape are the release gate in
//! tests/prefill_serve_bench.rs.

#![cfg(feature = "cuda")]

#[path = "../../tests/common/stamp.rs"]
mod stamp;
#[path = "../../tests/common/timed.rs"]
mod timed;
use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::backward::GpuMambaTargetScratch;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::forward::GpuMambaDims;
use mamba_rs::mamba_ssm::gpu::inference::GpuInferenceState;
use mamba_rs::mamba_ssm::gpu::prefill::{
    PrefillOutputs, PrefillPooledGraph, PrefillRawInputs, gpu_forward_inference_prefill_from_raw,
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
    }
}

fn set_tier(ctx: &GpuCtx, tier: &str) {
    match tier {
        "cublas+tf32" => {
            ctx.set_gemm_mode(GemmMode::CublasPedantic).unwrap();
            ctx.set_bi_tensor_cores(false);
            ctx.set_gemm_mode(GemmMode::CublasPedantic).unwrap();
        }
        "cublas" => {
            ctx.set_bi_tensor_cores(false);
            ctx.set_gemm_mode(GemmMode::CublasPedantic).unwrap();
            ctx.set_gemm_mode(GemmMode::CublasPedantic).unwrap();
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

/// I-5b: the headline serve numbers on the pinned serve tier —
/// full-temporal eager (the eval/calibrate lane), pooled eager, and the
/// production pooled graph. ms/page and pages/s.
#[test]
#[ignore = "serve headline bench"]
fn prefill_serve_headline() {
    let mut r = rig(16);
    set_tier(&r.ctx, "cublas+tf32");
    let dm = r.dims.d_model;

    let mut last = GpuBuffer::zeros(&r.ctx.stream, dm).unwrap();
    let mut full = GpuBuffer::zeros(&r.ctx.stream, SERVE_T * dm).unwrap();
    let mut host_full = vec![0f32; SERVE_T * dm];
    let full_eager_ms = {
        let ctx = &r.ctx;
        let state = &mut r.state;
        let scratch = &mut r.scratch;
        let input = &r.input;
        let weights = &r.weights;
        let a_neg = &r.a_neg;
        let mut run = || {
            state.reset(&ctx.stream).unwrap();
            gpu_forward_inference_prefill_from_raw(
                ctx,
                PrefillOutputs {
                    last_temporal: &mut last,
                    full_temporal: Some(&mut full),
                },
                PrefillRawInputs {
                    input_flat: input,
                    weights,
                    a_neg_all: a_neg,
                },
                state,
                scratch,
            )
            .unwrap();
            full.download(&ctx.stream, &mut host_full).unwrap();
        };
        timed::timed(ctx, 50, &mut run)
    };

    let mut pooled = GpuBuffer::zeros(&r.ctx.stream, dm).unwrap();
    let mut host_pooled = vec![0f32; dm];
    let pooled_eager_ms = {
        let ctx = &r.ctx;
        let state = &mut r.state;
        let scratch = &mut r.scratch;
        let input = &r.input;
        let weights = &r.weights;
        let a_neg = &r.a_neg;
        let mut run = || {
            state.reset(&ctx.stream).unwrap();
            gpu_forward_inference_prefill_pooled_sum_from_raw(
                ctx,
                &mut pooled,
                PrefillRawInputs {
                    input_flat: input,
                    weights,
                    a_neg_all: a_neg,
                },
                state,
                scratch,
            )
            .unwrap();
            pooled.download(&ctx.stream, &mut host_pooled).unwrap();
        };
        timed::timed(ctx, 50, &mut run)
    };

    // The benchmark keeps every captured allocation alive through the graph.
    let graph = unsafe {
        PrefillPooledGraph::capture(
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
    }
    .unwrap();
    let pooled_graph_ms = {
        let ctx = &r.ctx;
        let mut run = || {
            graph.launch(ctx, &r.input, &pooled).unwrap();
            pooled.download(&ctx.stream, &mut host_pooled).unwrap();
        };
        timed::timed(ctx, 50, &mut run)
    };

    eprintln!(
        "{}",
        stamp::bench_stamp(
            &r.device,
            &r.ctx,
            &format!("serve B1 T{SERVE_T} d384 L24"),
            "serve_headline",
            0
        )
    );
    eprintln!(
        "serve headline: full_eager={full_eager_ms:.3} ms/page ({:.1} pages/s)  \
         pooled_eager={pooled_eager_ms:.3} ms/page ({:.1} pages/s)  \
         pooled_graph={pooled_graph_ms:.3} ms/page ({:.1} pages/s)",
        1e3 / full_eager_ms,
        1e3 / pooled_eager_ms,
        1e3 / pooled_graph_ms
    );
}
