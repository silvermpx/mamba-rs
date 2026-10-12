//! Pooled-prefill oracle: the on-device pooled-sum prefill must reproduce
//! the full-temporal download + host column-sum BIT-FOR-BIT (the kernel
//! walks ascending t accumulating pure f32 adds from 0.0 — the same
//! reduction order as a per-column host loop), in a batch each sequence's
//! row is the one it gets alone, the raw-input entries take half input
//! projections and models without one, and the pinned staging buffer must
//! be transfer-transparent (same bytes as a pageable Vec).

#![cfg(feature = "cuda")]

use mamba_rs::gpu::GemmMode;
use mamba_rs::gpu::blas::gpu_gemm_forward_dispatch;
use mamba_rs::gpu::buffers::{GpuBuffer, PinnedHostBuf};
use mamba_rs::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba1::MambaWeights;
use mamba_rs::mamba1::gpu::backward::GpuMambaTargetScratch;
use mamba_rs::mamba1::gpu::forward::GpuMambaDims;
use mamba_rs::mamba1::gpu::inference::MambaGpuState;
use mamba_rs::mamba1::gpu::prefill::{
    PrefillInputs, PrefillOutputs, PrefillPooledGraph, PrefillRawInputs,
    gpu_forward_inference_prefill_from_raw, gpu_forward_inference_prefill_full,
    gpu_forward_inference_prefill_pooled_sum_from_raw,
};
use mamba_rs::mamba1::gpu::weights::{GpuMambaMixedWeights, GpuMambaWeights, MambaWeightsView};
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

struct Rig {
    ctx: GpuCtx,
    weights: GpuMambaWeights,
    scratch: GpuMambaTargetScratch,
    state: MambaGpuState,
    a_neg: GpuBuffer,
    input: GpuBuffer,
    dims: GpuMambaDims,
}

fn rig(seq_len: usize) -> Rig {
    rig_batch(seq_len, 1)
}

fn rig_cfg() -> MambaConfig {
    MambaConfig {
        d_model: 48,
        n_layers: 2,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    }
}

/// The rig's host weights, a_neg filled.
fn rig_weights(cfg: &MambaConfig, mid: usize) -> MambaWeights {
    let mut w = MambaWeights::init(cfg, mid, 0xBEEF);
    for lw in w.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    w
}

fn rig_batch(seq_len: usize, batch: usize) -> Rig {
    rig_with(seq_len, batch, 96, false)
}

/// A rig of `batch` sequences of `seq_len` raw rows of `mid`; `identity`
/// drops the input projection (then `mid` must be d_model).
fn rig_with(seq_len: usize, batch: usize, mid: usize, identity: bool) -> Rig {
    let cfg = rig_cfg();
    let mut w = rig_weights(&cfg, mid);
    if identity {
        w.input_proj_w.clear();
        w.input_proj_b.clear();
    }
    let (di, ds, nl) = (cfg.d_inner(), cfg.d_state, cfg.n_layers);
    let dims = GpuMambaDims {
        batch,
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
    let state = MambaGpuState::zeros(&ctx.stream, batch, &cfg).expect("state");
    let mut a_neg_flat = vec![0.0f32; nl * di * ds];
    for (l, lw) in w.layers.iter().enumerate() {
        a_neg_flat[l * di * ds..(l + 1) * di * ds].copy_from_slice(&lw.a_neg);
    }
    let mut a_neg = GpuBuffer::zeros(&ctx.stream, nl * di * ds).unwrap();
    a_neg.upload(&ctx.stream, &a_neg_flat).unwrap();
    let host_input = det(batch * seq_len * mid, 0x51, 0.05);
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

/// A batch of three: each sequence's pooled row is the host column sum of
/// its own rows of the full temporal, and the bits it gets alone.
#[test]
fn pooled_sum_of_a_batch_is_each_sequence_alone() {
    let batch = 3usize;
    for t in [64usize, 333] {
        let mut r = rig_batch(t, batch);
        let (dm, mid) = (r.dims.d_model, r.dims.mamba_input_dim);
        let mut last = GpuBuffer::zeros(&r.ctx.stream, batch * dm).unwrap();
        let mut full = GpuBuffer::zeros(&r.ctx.stream, batch * t * dm).unwrap();
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
        let mut temporal = vec![0.0f32; batch * t * dm];
        full.download(&r.ctx.stream, &mut temporal).unwrap();
        let mut pooled = GpuBuffer::zeros(&r.ctx.stream, batch * dm).unwrap();
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
        let mut device_sum = vec![0.0f32; batch * dm];
        pooled.download(&r.ctx.stream, &mut device_sum).unwrap();
        let host_input = det(batch * t * mid, 0x51, 0.05);
        for s in 0..batch {
            let mut host_sum = vec![0.0f32; dm];
            for (j, acc) in host_sum.iter_mut().enumerate() {
                for row in 0..t {
                    *acc += temporal[(s * t + row) * dm + j];
                }
            }
            let mut one = rig(t);
            one.input
                .upload(&one.ctx.stream, &host_input[s * t * mid..(s + 1) * t * mid])
                .unwrap();
            let mut alone = GpuBuffer::zeros(&one.ctx.stream, dm).unwrap();
            one.state.reset(&one.ctx.stream).unwrap();
            gpu_forward_inference_prefill_pooled_sum_from_raw(
                &one.ctx,
                &mut alone,
                PrefillRawInputs {
                    input_flat: &one.input,
                    weights: &one.weights,
                    a_neg_all: &one.a_neg,
                },
                &mut one.state,
                &mut one.scratch,
            )
            .expect("pooled prefill alone");
            let mut alone_sum = vec![0.0f32; dm];
            alone.download(&one.ctx.stream, &mut alone_sum).unwrap();
            let row = &device_sum[s * dm..(s + 1) * dm];
            for (j, ((d, h), a)) in row.iter().zip(&host_sum).zip(&alone_sum).enumerate() {
                assert_eq!(
                    d.to_bits(),
                    h.to_bits(),
                    "T={t} sequence {s} dim {j}: device {d} vs host {h}"
                );
                assert_eq!(
                    d.to_bits(),
                    a.to_bits(),
                    "T={t} sequence {s} dim {j}: in the batch {d} vs alone {a}"
                );
            }
        }
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The full rows and the last rows of one prefill, raw or projected.
fn full_and_last(r: &mut Rig, prefill: impl FnOnce(&mut Rig, PrefillOutputs<'_>)) -> Vec<f32> {
    let (b, t, dm) = (r.dims.batch, r.dims.seq_len, r.dims.d_model);
    let mut last = GpuBuffer::zeros(&r.ctx.stream, b * dm).unwrap();
    let mut full = GpuBuffer::zeros(&r.ctx.stream, b * t * dm).unwrap();
    r.state.reset(&r.ctx.stream).unwrap();
    prefill(
        r,
        PrefillOutputs {
            last_temporal: &mut last,
            full_temporal: Some(&mut full),
        },
    );
    let mut rows = vec![0.0f32; b * t * dm];
    full.download(&r.ctx.stream, &mut rows).unwrap();
    let mut tail = vec![0.0f32; b * dm];
    last.download(&r.ctx.stream, &mut tail).unwrap();
    rows.extend(tail);
    rows
}

/// The raw-input entries take a half input projection: the rows cast to
/// the weights' dtype through the GEMM the engine's prompts take, so a raw
/// prefill gives the bits of one handed that projection, and its pooled
/// sum is the host column sum of its rows.
#[test]
fn raw_input_takes_a_half_input_projection() {
    let t = 333usize;
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let mut r = rig(t);
        let cfg = rig_cfg();
        let (dm, mid) = (r.dims.d_model, r.dims.mamba_input_dim);
        let w = rig_weights(&cfg, mid);
        let half = GpuMambaMixedWeights::from_cpu(&r.ctx.stream, &w, &cfg, dtype).unwrap();
        let mut projected = GpuBuffer::zeros(&r.ctx.stream, t * dm).unwrap();
        let (ipw, ipw_dt) = half.input_proj_w();
        gpu_gemm_forward_dispatch(
            &r.ctx,
            &mut projected,
            &r.input,
            ipw,
            ipw_dt,
            Some(half.input_proj_b()),
            (t, mid, dm),
        )
        .unwrap();
        let want = full_and_last(&mut r, |r, outputs| {
            gpu_forward_inference_prefill_full(
                &r.ctx,
                outputs,
                PrefillInputs {
                    ip_out_flat: &projected,
                    weights: &half,
                    a_neg_all: &r.a_neg,
                },
                &mut r.state,
                &mut r.scratch,
            )
            .expect("projected prefill");
        });
        let got = full_and_last(&mut r, |r, outputs| {
            gpu_forward_inference_prefill_from_raw(
                &r.ctx,
                outputs,
                PrefillRawInputs {
                    input_flat: &r.input,
                    weights: &half,
                    a_neg_all: &r.a_neg,
                },
                &mut r.state,
                &mut r.scratch,
            )
            .expect("raw prefill");
        });
        assert!(
            bits(&got) == bits(&want),
            "{dtype:?}: the raw prefill moved off the projected one"
        );
        let mut pooled = GpuBuffer::zeros(&r.ctx.stream, dm).unwrap();
        r.state.reset(&r.ctx.stream).unwrap();
        gpu_forward_inference_prefill_pooled_sum_from_raw(
            &r.ctx,
            &mut pooled,
            PrefillRawInputs {
                input_flat: &r.input,
                weights: &half,
                a_neg_all: &r.a_neg,
            },
            &mut r.state,
            &mut r.scratch,
        )
        .expect("pooled prefill");
        let mut device_sum = vec![0.0f32; dm];
        pooled.download(&r.ctx.stream, &mut device_sum).unwrap();
        for (j, d) in device_sum.iter().enumerate() {
            let h = (0..t).fold(0.0f32, |acc, row| acc + got[row * dm + j]);
            assert_eq!(
                d.to_bits(),
                h.to_bits(),
                "{dtype:?} dim {j}: device {d} vs host {h}"
            );
        }
    }
}

/// A model without an input projection takes its rows as they are: the
/// bits of a prefill handed those rows.
#[test]
fn raw_input_without_an_input_projection_is_the_rows() {
    let t = 64usize;
    let dm = rig_cfg().d_model;
    let mut r = rig_with(t, 2, dm, true);
    assert!(!r.weights.has_input_proj());
    let rows = det(2 * t * dm, 0x51, 0.05);
    let mut given = GpuBuffer::zeros(&r.ctx.stream, rows.len()).unwrap();
    given.upload(&r.ctx.stream, &rows).unwrap();
    let want = full_and_last(&mut r, |r, outputs| {
        gpu_forward_inference_prefill_full(
            &r.ctx,
            outputs,
            PrefillInputs {
                ip_out_flat: &given,
                weights: &r.weights,
                a_neg_all: &r.a_neg,
            },
            &mut r.state,
            &mut r.scratch,
        )
        .expect("prefill of the rows");
    });
    let got = full_and_last(&mut r, |r, outputs| {
        gpu_forward_inference_prefill_from_raw(
            &r.ctx,
            outputs,
            PrefillRawInputs {
                input_flat: &r.input,
                weights: &r.weights,
                a_neg_all: &r.a_neg,
            },
            &mut r.state,
            &mut r.scratch,
        )
        .expect("raw prefill");
    });
    assert!(
        bits(&got) == bits(&want),
        "the raw rows were not taken as they are"
    );
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

/// The captured graph replays the exact kernel sequence: replay output ==
/// eager output bitwise, and 100 replays of the same page are all stable.
#[test]
fn pooled_graph_replays_bitwise() {
    for family in [BiGemmFamily::Triad, BiGemmFamily::Inference] {
        pooled_graph_replays_bitwise_for_family(family);
    }
}

fn pooled_graph_replays_bitwise_for_family(family: BiGemmFamily) {
    let t = 333usize;
    let mut r = rig(t);
    r.ctx.route_controls().set_family(family);
    let dm = r.dims.d_model;
    // Eager reference.
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
    .expect("eager pooled");
    let mut eager = vec![0.0f32; dm];
    pooled.download(&r.ctx.stream, &mut eager).unwrap();
    // Capture (runs the body once as part of capture) then replay 100x.
    // The rig and pooled buffer outlive the graph in this scope.
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
    .expect("capture");
    let other_device = GpuDevice::new(0).unwrap();
    let other_ctx = GpuCtx::new(&other_device).unwrap();
    let error = graph
        .launch(&other_ctx, &r.input, &pooled)
        .expect_err("launch must reject a different GpuCtx");
    assert!(error.contains("GpuCtx differs from capture"));
    for rep in 0..100 {
        graph.launch(&r.ctx, &r.input, &pooled).expect("replay");
        let mut got = vec![0.0f32; dm];
        pooled.download(&r.ctx.stream, &mut got).unwrap();
        for (j, (a, b)) in eager.iter().zip(&got).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "replay {rep} dim {j}: eager {a} vs graph {b}"
            );
        }
    }
}

/// A captured pooled graph must refuse to replay after ANY numeric-route
/// policy flip - the captured kernels belong to the route at capture.
#[test]
fn pooled_graph_refuses_route_drift() {
    let mut r = rig(192);
    let dm = r.dims.d_model;
    let mut pooled = GpuBuffer::zeros(&r.ctx.stream, dm).unwrap();
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
    .expect("capture");

    type RouteFlip<'a> = (&'a str, &'a dyn Fn(), &'a dyn Fn());
    let flips: &[RouteFlip<'_>] = &[
        (
            "cublas_pedantic",
            &|| r.ctx.set_gemm_mode(GemmMode::CublasPedantic).unwrap(),
            &|| r.ctx.set_gemm_mode(GemmMode::Deterministic).unwrap(),
        ),
        (
            "cublas_fast",
            &|| r.ctx.set_gemm_mode(GemmMode::CublasFast).unwrap(),
            &|| r.ctx.set_gemm_mode(GemmMode::Deterministic).unwrap(),
        ),
        (
            "family",
            &|| {
                r.ctx
                    .route_controls()
                    .set_family(mamba_rs::gpu::context::BiGemmFamily::Inference)
            },
            &|| {
                r.ctx
                    .route_controls()
                    .set_family(mamba_rs::gpu::context::BiGemmFamily::Triad)
            },
        ),
    ];
    for (name, flip, restore) in flips {
        flip();
        let err = graph
            .launch(&r.ctx, &r.input, &pooled)
            .expect_err("route flip must refuse replay");
        assert!(err.contains("GEMM flags"), "{name}: {err}");
        restore();
    }
    // A vendor mode left in place must keep refusing the replay.
    r.ctx.set_gemm_mode(GemmMode::CublasPedantic).unwrap();
    let err = graph
        .launch(&r.ctx, &r.input, &pooled)
        .expect_err("mode flip must refuse replay");
    assert!(err.contains("GEMM flags"), "gemm_mode: {err}");
}

/// The buffer pins: replaying against a different pooled buffer must
/// refuse (the graph writes the captured allocation, not the argument).
#[test]
fn pooled_graph_refuses_swapped_buffers() {
    let mut r = rig(192);
    let dm = r.dims.d_model;
    let mut pooled = GpuBuffer::zeros(&r.ctx.stream, dm).unwrap();
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
    .expect("capture");
    let other = GpuBuffer::zeros(&r.ctx.stream, dm).unwrap();
    let err = graph
        .launch(&r.ctx, &r.input, &other)
        .expect_err("swapped pooled buffer must refuse");
    assert!(err.contains("captured allocations"), "{err}");
    graph
        .launch(&r.ctx, &r.input, &pooled)
        .expect("real buffers replay");
}
