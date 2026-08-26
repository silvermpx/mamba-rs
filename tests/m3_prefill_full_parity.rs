//! The m3 serve-surface acceptance gate: `Mamba3Prefill::run_full` must
//! reproduce the trainer forward's post-norm_f temporal BIT FOR BIT (the
//! classify head and its calibration are fitted on the trainer-forward
//! temporal - a serve surface that diverges by one ulp would score through
//! a numeric route the temperature was never fitted on), the pooled column
//! sum must equal the ascending-t host fold of that temporal exactly, and
//! the pooled CUDA-graph replay must equal the eager pooled path.
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::trainer::TrainSessionCfg;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
use mamba_rs::mamba3_siso::gpu::prefill::{
    Mamba3Prefill, Mamba3PrefillOutputs, Mamba3PrefillPooledGraph, Mamba3PrefillRun,
};
use mamba_rs::mamba3_siso::gpu::state::{GpuMamba3Dims, GpuMamba3StateBufs};
use mamba_rs::mamba3_siso::gpu::trainer::Mamba3Trainer;
use mamba_rs::mamba3_siso::gpu::weights::GpuMamba3WeightsInf;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn tiny_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 1e-4,
        is_outproj_norm: false,
        ..Mamba3Config::default()
    }
}

fn gpu_dims(cfg: &Mamba3Config, input_dim: usize, seq_len: usize) -> GpuMamba3Dims {
    GpuMamba3Dims {
        batch: 1,
        d_model: cfg.d_model,
        d_inner: cfg.d_inner(),
        d_state: cfg.d_state,
        nheads: cfg.nheads(),
        headdim: cfg.headdim,
        ngroups: cfg.ngroups,
        in_proj_dim: cfg.in_proj_out_dim(),
        seq_len,
        mamba_input_dim: input_dim,
        n_layers: cfg.n_layers,
        n_angles: cfg.num_rope_angles(),
        a_floor: cfg.a_floor,
        is_outproj_norm: cfg.is_outproj_norm,
        rms_norm_eps: cfg.rms_norm_eps,
        use_parallel_scan: true,
    }
}

fn det_input(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0
        })
        .collect()
}

fn assert_bitwise(tag: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{tag}: length");
    let bad = got
        .iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(bad, 0, "{tag}: {bad}/{} values differ bitwise", got.len());
}

/// Full-T serve surface == trainer forward, bit for bit; pooled == the
/// ascending-t host fold; last_hidden == the temporal's final row; the
/// pooled graph replay == the eager pooled path.
#[test]
fn m3_prefill_full_matches_trainer_forward() {
    let cfg = tiny_cfg();
    let input_dim = 24usize;
    let seq_len = 192usize;
    let dm = cfg.d_model;
    let w = Mamba3Weights::init(&cfg, input_dim, 0x00C0_FFEE);
    let input = det_input(seq_len * input_dim, 77);

    // Trainer-forward reference temporal (the surface the classify lane
    // scores through today).
    let mut trainer = Mamba3Trainer::new_full(
        0,
        &w,
        cfg,
        TrainSessionCfg {
            input_dim,
            batch: 1,
            seq_len,
            lr: 1e-5,
            weight_decay: 0.0,
        },
        WeightDtype::F32,
    )
    .expect("m3 f32 trainer");
    let mut reference = vec![0.0f32; seq_len * dm];
    trainer.reset_state().unwrap();
    trainer.forward(&input, &mut reference).unwrap();

    // Serve surface on a fresh context.
    let device = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&device).expect("ctx");
    let arch = GpuDevice::nvrtc_arch(device.compute_capability);
    let kernels = Mamba3Kernels::compile(device.context(), arch).expect("m3 kernels");
    let dims = gpu_dims(&cfg, input_dim, seq_len);
    let gw = GpuMamba3WeightsInf::from_cpu(&ctx.stream, &w, input_dim).unwrap();
    let mut gpu_input = GpuBuffer::from_cpu(&ctx.stream, &input).unwrap();
    let mut prefill = Mamba3Prefill::new(&ctx.stream, &dims).unwrap();
    let mut last_hidden = GpuBuffer::zeros(&ctx.stream, dm).unwrap();
    let mut full = GpuBuffer::zeros(&ctx.stream, seq_len * dm).unwrap();
    let mut pooled = GpuBuffer::zeros(&ctx.stream, dm).unwrap();
    let nl = cfg.n_layers;
    let mut ssm =
        GpuBuffer::zeros(&ctx.stream, nl * cfg.nheads() * cfg.headdim * cfg.d_state).unwrap();
    let mut kst = GpuBuffer::zeros(&ctx.stream, nl * cfg.nheads() * cfg.d_state).unwrap();
    let mut vst = GpuBuffer::zeros(&ctx.stream, nl * cfg.nheads() * cfg.headdim).unwrap();
    let mut ast = GpuBuffer::zeros(
        &ctx.stream,
        nl * cfg.nheads() * cfg.num_rope_angles().max(1),
    )
    .unwrap();

    prefill
        .run_full(
            &Mamba3PrefillRun {
                ctx: &ctx,
                kernels: &kernels,
                dims: &dims,
                weights: &gw,
                mamba_input: &gpu_input,
                identity_proj: false,
                carry_state: false,
            },
            GpuMamba3StateBufs {
                ssm: &mut ssm,
                k: &mut kst,
                v: &mut vst,
                angle: &mut ast,
            },
            Mamba3PrefillOutputs {
                last_hidden: &mut last_hidden,
                full_temporal: Some(&mut full),
                pooled_sum: Some(&mut pooled),
            },
        )
        .unwrap();
    ctx.stream.synchronize().unwrap();

    let mut full_h = vec![0.0f32; seq_len * dm];
    full.download(&ctx.stream, &mut full_h).unwrap();
    let mut pooled_h = vec![0.0f32; dm];
    pooled.download(&ctx.stream, &mut pooled_h).unwrap();
    let mut last_h = vec![0.0f32; dm];
    last_hidden.download(&ctx.stream, &mut last_h).unwrap();

    assert_bitwise("full_temporal vs trainer forward", &full_h, &reference);
    assert_bitwise(
        "last_hidden vs final temporal row",
        &last_h,
        &full_h[(seq_len - 1) * dm..],
    );

    // Pooled contract: ascending-t f32 column sums, division on the host.
    let mut host_sum = vec![0.0f32; dm];
    for row in full_h.chunks(dm) {
        for (acc, &v) in host_sum.iter_mut().zip(row) {
            *acc += v;
        }
    }
    assert_bitwise("pooled_sum vs ascending-t host fold", &pooled_h, &host_sum);

    // Graph replay: same buffers, next page - bit-equal to eager.
    // Every captured allocation remains alive through graph destruction.
    let graph = unsafe {
        Mamba3PrefillPooledGraph::capture(
            &mut prefill,
            &Mamba3PrefillRun {
                ctx: &ctx,
                kernels: &kernels,
                dims: &dims,
                weights: &gw,
                mamba_input: &gpu_input,
                identity_proj: false,
                carry_state: false,
            },
            GpuMamba3StateBufs {
                ssm: &mut ssm,
                k: &mut kst,
                v: &mut vst,
                angle: &mut ast,
            },
            &mut last_hidden,
            &mut pooled,
        )
    }
    .unwrap();
    let page2 = det_input(seq_len * input_dim, 78);
    gpu_input.upload(&ctx.stream, &page2).unwrap();
    let other_ctx = GpuCtx::new(&device).unwrap();
    let error = graph
        .replay(
            &other_ctx,
            &kernels,
            &gw,
            &gpu_input,
            &GpuMamba3StateBufs {
                ssm: &mut ssm,
                k: &mut kst,
                v: &mut vst,
                angle: &mut ast,
            },
            &pooled,
        )
        .expect_err("replay must reject a different GpuCtx");
    assert!(error.contains("GpuCtx differs from capture"));
    graph
        .replay(
            &ctx,
            &kernels,
            &gw,
            &gpu_input,
            &GpuMamba3StateBufs {
                ssm: &mut ssm,
                k: &mut kst,
                v: &mut vst,
                angle: &mut ast,
            },
            &pooled,
        )
        .unwrap();
    ctx.stream.synchronize().unwrap();
    let mut pooled_graph = vec![0.0f32; dm];
    pooled.download(&ctx.stream, &mut pooled_graph).unwrap();

    // Eager pooled on the same second page for the comparison.
    prefill
        .run_full(
            &Mamba3PrefillRun {
                ctx: &ctx,
                kernels: &kernels,
                dims: &dims,
                weights: &gw,
                mamba_input: &gpu_input,
                identity_proj: false,
                carry_state: false,
            },
            GpuMamba3StateBufs {
                ssm: &mut ssm,
                k: &mut kst,
                v: &mut vst,
                angle: &mut ast,
            },
            Mamba3PrefillOutputs {
                last_hidden: &mut last_hidden,
                full_temporal: None,
                pooled_sum: Some(&mut pooled),
            },
        )
        .unwrap();
    ctx.stream.synchronize().unwrap();
    let mut pooled_eager = vec![0.0f32; dm];
    pooled.download(&ctx.stream, &mut pooled_eager).unwrap();
    assert_bitwise("pooled graph replay vs eager", &pooled_graph, &pooled_eager);
}
