//! parity test for M3 mixed forward (sequential SSM only) vs f32
//! backbone. Validates that the typed kernel wiring produces outputs within
//! bf16/f16 ULP cosine of the f32 oracle.

#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::forward_mixed::{
    GpuMamba3BackboneMixedActs, GpuMamba3MixedScratch, gpu_forward_mamba3_backbone_mixed,
};
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
use mamba_rs::mamba3_siso::gpu::mamba3_gpu::{
    GpuMamba3BackboneActs, GpuMamba3Dims, GpuMamba3Scratch, GpuMamba3StateBufs, M3Exec,
    gpu_forward_mamba3_backbone,
};
use mamba_rs::mamba3_siso::gpu::weights::GpuMamba3Weights;
use mamba_rs::mamba3_siso::gpu::weights_mixed_train::GpuMamba3TrainMixedWeights;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn tiny_cfg() -> Mamba3Config {
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
        ..Mamba3Config::default()
    }
}

fn dims_for(cfg: &Mamba3Config, batch: usize, seq_len: usize) -> GpuMamba3Dims {
    GpuMamba3Dims {
        batch,
        d_model: cfg.d_model,
        d_inner: cfg.d_inner(),
        d_state: cfg.d_state,
        nheads: cfg.nheads(),
        headdim: cfg.headdim,
        ngroups: cfg.ngroups,
        in_proj_dim: cfg.in_proj_out_dim(),
        seq_len,
        mamba_input_dim: cfg.d_model,
        n_layers: cfg.n_layers,
        n_angles: cfg.num_rope_angles(),
        a_floor: cfg.a_floor,
        rms_norm_eps: cfg.rms_norm_eps,
        is_outproj_norm: cfg.is_outproj_norm,
        use_parallel_scan: false,
    }
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

fn cos_sim(a: &[f32], b: &[f32]) -> (f32, f32) {
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
    cpu: &Mamba3Weights,
    cfg: &Mamba3Config,
    dims: &GpuMamba3Dims,
    input: &[f32],
) -> Vec<f32> {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = Mamba3Kernels::compile(dev.context(), arch::arch0()).unwrap();
    let w = GpuMamba3Weights::from_cpu(&ctx.stream, cpu, cfg, dims.mamba_input_dim).unwrap();

    let bt = dims.bt();
    let (nh, hd, ds, na) = (
        dims.nheads,
        dims.headdim,
        dims.d_state,
        dims.n_angles.max(1),
    );
    let b = dims.batch;
    let nl = dims.n_layers;
    let mut ssm = GpuBuffer::zeros(&ctx.stream, b * nl * nh * hd * ds).unwrap();
    let mut ks = GpuBuffer::zeros(&ctx.stream, b * nl * nh * ds).unwrap();
    let mut vs = GpuBuffer::zeros(&ctx.stream, b * nl * nh * hd).unwrap();
    let mut ang = GpuBuffer::zeros(&ctx.stream, b * nl * nh * na).unwrap();
    ctx.stream.synchronize().unwrap();

    let mut mi = GpuBuffer::zeros(&ctx.stream, bt * dims.mamba_input_dim).unwrap();
    ctx.stream.synchronize().unwrap();
    mi.upload(&ctx.stream, input).unwrap();

    let mut temporal = GpuBuffer::zeros(&ctx.stream, bt * dims.d_model).unwrap();
    let mut acts = GpuMamba3BackboneActs::new(&ctx.stream, dims).unwrap();
    let mut scratch = GpuMamba3Scratch::new(&ctx.stream, dims).unwrap();
    ctx.stream.synchronize().unwrap();

    gpu_forward_mamba3_backbone(
        &M3Exec {
            ctx: &ctx,
            kernels: &m3k,
            dims,
        },
        &mut temporal,
        &mut acts,
        &w,
        &mi,
        GpuMamba3StateBufs {
            ssm: &mut ssm,
            k: &mut ks,
            v: &mut vs,
            angle: &mut ang,
        },
        &mut scratch,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();

    let mut out = vec![0f32; bt * dims.d_model];
    temporal.download(&ctx.stream, &mut out).unwrap();
    out
}

fn run_mixed(
    cpu: &Mamba3Weights,
    cfg: &Mamba3Config,
    dims: &GpuMamba3Dims,
    dtype: WeightDtype,
    input: &[f32],
) -> Vec<f32> {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = Mamba3Kernels::compile(dev.context(), arch::arch0()).unwrap();

    // Mixed path needs identity input_proj — clone + clear.
    let mut cpu_m = cpu.clone();
    cpu_m.input_proj_w.clear();
    cpu_m.input_proj_b.clear();
    let w =
        GpuMamba3TrainMixedWeights::from_cpu(&ctx.stream, &cpu_m, cfg, dims.mamba_input_dim, dtype)
            .unwrap();

    let bt = dims.bt();
    let (nh, hd, ds, na) = (
        dims.nheads,
        dims.headdim,
        dims.d_state,
        dims.n_angles.max(1),
    );
    let b = dims.batch;
    let nl = dims.n_layers;
    let mut ssm = GpuBuffer::zeros(&ctx.stream, b * nl * nh * hd * ds).unwrap();
    let mut ks = GpuBuffer::zeros(&ctx.stream, b * nl * nh * ds).unwrap();
    let mut vs = GpuBuffer::zeros(&ctx.stream, b * nl * nh * hd).unwrap();
    let mut ang = GpuBuffer::zeros(&ctx.stream, b * nl * nh * na).unwrap();
    ctx.stream.synchronize().unwrap();

    let mut mi = GpuBuffer::zeros(&ctx.stream, bt * dims.mamba_input_dim).unwrap();
    ctx.stream.synchronize().unwrap();
    mi.upload(&ctx.stream, input).unwrap();

    let mut temporal = GpuBuffer::zeros(&ctx.stream, bt * dims.d_model).unwrap();
    let mut acts = GpuMamba3BackboneMixedActs::new(
        &ctx.stream,
        cfg,
        b,
        dims.seq_len,
        dims.mamba_input_dim,
        dtype,
        dims.use_parallel_scan,
    )
    .unwrap();
    let mut scratch = GpuMamba3MixedScratch::new(&ctx.stream, cfg, b, dims.seq_len, dtype).unwrap();
    ctx.stream.synchronize().unwrap();

    gpu_forward_mamba3_backbone_mixed(
        &M3Exec {
            ctx: &ctx,
            kernels: &m3k,
            dims,
        },
        &mut temporal,
        &mut acts,
        &w,
        &mi,
        GpuMamba3StateBufs {
            ssm: &mut ssm,
            k: &mut ks,
            v: &mut vs,
            angle: &mut ang,
        },
        &mut scratch,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();

    // `temporal_f32` holds the POST-norm_f output — the public contract
    // the classifier head consumes. Comparing this surface (not the
    // pre-norm residual) is what catches a dropped final norm: the
    // pre-norm buffers of the two lanes can agree while the head sees a
    // completely different stream.
    let mut out = vec![0f32; bt * dims.d_model];
    temporal.download(&ctx.stream, &mut out).unwrap();
    out
}

/// f32-reference path returning the POST-norm_f `temporal` — the same
/// public surface the mixed lane must produce.
fn run_f32_post_normf(
    cpu: &Mamba3Weights,
    cfg: &Mamba3Config,
    dims: &GpuMamba3Dims,
    input: &[f32],
) -> Vec<f32> {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = Mamba3Kernels::compile(dev.context(), arch::arch0()).unwrap();
    // F32 forward has no identity-proj branch; use eye(d_model) + zero bias
    // so `temporal = mamba_input @ I = mamba_input` matches mixed's D2D copy.
    let mut cpu_m = cpu.clone();
    cpu_m.input_proj_w = (0..cfg.d_model * cfg.d_model)
        .map(|i| {
            let r = i / cfg.d_model;
            let c = i % cfg.d_model;
            if r == c { 1.0 } else { 0.0 }
        })
        .collect();
    cpu_m.input_proj_b = vec![0.0; cfg.d_model];
    let w = GpuMamba3Weights::from_cpu(&ctx.stream, &cpu_m, cfg, dims.mamba_input_dim).unwrap();

    let bt = dims.bt();
    let (nh, hd, ds, na) = (
        dims.nheads,
        dims.headdim,
        dims.d_state,
        dims.n_angles.max(1),
    );
    let b = dims.batch;
    let nl = dims.n_layers;
    let mut ssm = GpuBuffer::zeros(&ctx.stream, b * nl * nh * hd * ds).unwrap();
    let mut ks = GpuBuffer::zeros(&ctx.stream, b * nl * nh * ds).unwrap();
    let mut vs = GpuBuffer::zeros(&ctx.stream, b * nl * nh * hd).unwrap();
    let mut ang = GpuBuffer::zeros(&ctx.stream, b * nl * nh * na).unwrap();
    ctx.stream.synchronize().unwrap();
    let mut mi = GpuBuffer::zeros(&ctx.stream, bt * dims.mamba_input_dim).unwrap();
    ctx.stream.synchronize().unwrap();
    mi.upload(&ctx.stream, input).unwrap();
    let mut temporal = GpuBuffer::zeros(&ctx.stream, bt * dims.d_model).unwrap();
    let mut acts = GpuMamba3BackboneActs::new(&ctx.stream, dims).unwrap();
    let mut scratch = GpuMamba3Scratch::new(&ctx.stream, dims).unwrap();
    ctx.stream.synchronize().unwrap();
    gpu_forward_mamba3_backbone(
        &M3Exec {
            ctx: &ctx,
            kernels: &m3k,
            dims,
        },
        &mut temporal,
        &mut acts,
        &w,
        &mi,
        GpuMamba3StateBufs {
            ssm: &mut ssm,
            k: &mut ks,
            v: &mut vs,
            angle: &mut ang,
        },
        &mut scratch,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    // Return the post-norm_f `temporal` — the public output surface.
    let mut out = vec![0f32; bt * dims.d_model];
    temporal.download(&ctx.stream, &mut out).unwrap();
    out
}

fn check(dtype: WeightDtype) {
    let cfg = tiny_cfg();
    let dims = dims_for(&cfg, 1, 4);
    let cpu = Mamba3Weights::init(&cfg, dims.mamba_input_dim, 0xBADF00D8);
    let input = det_rand(dims.bt() * dims.mamba_input_dim, 0xA1);

    let ref_out = run_f32_post_normf(&cpu, &cfg, &dims, &input);
    let _unused = run_f32(&cpu, &cfg, &dims, &input); // exercise full f32 path to smoke it
    let typ_out = run_mixed(&cpu, &cfg, &dims, dtype, &input);

    let (cos, ratio) = cos_sim(&ref_out, &typ_out);
    eprintln!(
        "m3_fwd_mixed {dtype:?}: cos={cos:.6} norm_ratio={ratio:.4} n={}",
        ref_out.len()
    );
    let (cos_min, norm_tol) = match dtype {
        WeightDtype::Bf16 => (0.995_f32, 0.05_f32),
        WeightDtype::F16 => (0.999_f32, 0.02_f32),
        WeightDtype::F32 => unreachable!(),
    };
    assert!(cos >= cos_min, "{dtype:?}: cos {cos:.6} < {cos_min}");
    assert!(
        (ratio - 1.0).abs() <= norm_tol,
        "{dtype:?}: norm_ratio {ratio:.4} outside [1 ± {norm_tol}]"
    );
}

#[test]
fn m3_forward_mixed_parity_bf16() {
    check(WeightDtype::Bf16);
}

#[test]
fn m3_forward_mixed_parity_f16() {
    check(WeightDtype::F16);
}

/// The dropped-norm_f regression: the mixed forward's public output must
/// be the POST-norm stream. With norm scales at ones every output row has
/// RMS ~= 1 whatever the depth; the pre-norm residual of a deep stack
/// drifts far from 1, so a dropped final norm fails this immediately
/// (the defect shipped exactly that way: the head consumed the raw
/// 24-layer residual while the backward applied the norm VJP).
#[test]
fn m3_forward_mixed_output_is_post_norm() {
    let cfg = Mamba3Config {
        n_layers: 8,
        ..tiny_cfg()
    };
    let dims = dims_for(&cfg, 1, 4);
    let cpu = Mamba3Weights::init(&cfg, dims.mamba_input_dim, 0xBADF00D8);
    let input = det_rand(dims.bt() * dims.mamba_input_dim, 0xA1);
    let out = run_mixed(&cpu, &cfg, &dims, WeightDtype::Bf16, &input);
    for (r, row) in out.chunks(dims.d_model).enumerate() {
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32).sqrt();
        assert!(
            (0.8..=1.25).contains(&rms),
            "row {r}: output RMS {rms:.3} is not ~1 — the public surface is \
             not the post-norm_f stream"
        );
    }
}
