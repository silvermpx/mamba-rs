//! Step 11 — Mamba-3 SISO mixed-precision (bf16/f16) backward parity test.
//! Compares `gpu_backward_mamba3_backbone_mixed` (Step 10 typed bwd)
//! against the f32 oracle `gpu_backward_mamba3_backbone` on identical
//! inputs/weights, asserting tight cosine + norm-ratio thresholds across
//! the full f32 master-grad arena.
//!
//! This was the top release blocker per the 6-agent pre-release audit
//! (Agent 5 — GPU pipelines): Step 10 had unit parity for every typed
//! bwd kernel but no integration-level call site.
//!
//! ## Production-config required by Step 10
//!   - `use_parallel_scan = true` (chunked SSM bwd via Steps 9b + 9d)
//!   - `is_outproj_norm = true` (RMSNormGated via Step 9c)
//!   - `n_angles > 0` (RoPE via Step 9a)
//!   - `input_proj_w` identity (caller pre-clears in CPU weights)

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::backward::gpu_backward_mamba3_backbone;
use mamba_rs::mamba3_siso::gpu::backward_mixed::gpu_backward_mamba3_backbone_mixed;
use mamba_rs::mamba3_siso::gpu::forward::gpu_forward_mamba3_backbone;
use mamba_rs::mamba3_siso::gpu::forward_mixed::{
    GpuMamba3BackboneMixedActs, GpuMamba3MixedScratch, gpu_forward_mamba3_backbone_mixed,
};
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
use mamba_rs::mamba3_siso::gpu::state::{
    GpuMamba3BackboneActs, GpuMamba3Dims, GpuMamba3Scratch, GpuMamba3StateBufs, M3Exec,
};
use mamba_rs::mamba3_siso::gpu::weights::{GpuMamba3Grads, GpuMamba3Weights};
use mamba_rs::mamba3_siso::gpu::weights_mixed_train::GpuMamba3TrainMixedWeights;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn cfg_for_step10() -> Mamba3Config {
    Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5, // n_angles > 0
        a_floor: 0.0625,
        is_outproj_norm: true, // RMSNormGated path
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
        is_outproj_norm: cfg.is_outproj_norm,
        use_parallel_scan: true, // chunked path required for Step 10
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

fn cos_norm(a: &[f32], b: &[f32]) -> (f32, f32) {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
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

fn assert_close(label: &str, a: &[f32], b: &[f32], cos_min: f32, norm_tol: f32) {
    let (cos, ratio) = cos_norm(a, b);
    eprintln!("  {label}: cos={cos:.6} norm={ratio:.4} n={}", a.len());
    assert!(cos >= cos_min, "{label}: cos {cos} < {cos_min}");
    assert!(
        (ratio - 1.0).abs() <= norm_tol,
        "{label}: norm {ratio} outside [1 ± {norm_tol}]"
    );
}

/// Build CPU weights twice: once with eye(d_model) input_proj for the
/// f32 path (its forward has no identity branch), once with cleared
/// input_proj for the mixed path (its forward + backward both use the
/// identity branch via len_elems()==0). All other weights identical.
fn build_weights(cfg: &Mamba3Config, seed: u64) -> (Mamba3Weights, Mamba3Weights) {
    let w = Mamba3Weights::init(cfg, cfg.d_model, seed);
    let mut w_f32 = w.clone();
    w_f32.input_proj_w = (0..cfg.d_model * cfg.d_model)
        .map(|i| {
            let r = i / cfg.d_model;
            let c = i % cfg.d_model;
            if r == c { 1.0 } else { 0.0 }
        })
        .collect();
    w_f32.input_proj_b = vec![0.0; cfg.d_model];
    let mut w_mix = w;
    w_mix.input_proj_w.clear();
    w_mix.input_proj_b.clear();
    (w_f32, w_mix)
}

fn run_f32(
    exec: &M3Exec<'_>,
    cpu: &Mamba3Weights,
    cfg: &Mamba3Config,
    mamba_input: &[f32],
    d_temporal_init: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let M3Exec { ctx, dims, .. } = *exec;
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
    mi.upload(&ctx.stream, mamba_input).unwrap();

    let mut temporal = GpuBuffer::zeros(&ctx.stream, bt * dims.d_model).unwrap();
    let mut acts = GpuMamba3BackboneActs::new(&ctx.stream, dims).unwrap();
    let mut scratch = GpuMamba3Scratch::new(&ctx.stream, dims).unwrap();
    ctx.stream.synchronize().unwrap();

    gpu_forward_mamba3_backbone(
        exec,
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

    let mut d_temporal = GpuBuffer::zeros(&ctx.stream, bt * dims.d_model).unwrap();
    ctx.stream.synchronize().unwrap();
    d_temporal.upload(&ctx.stream, d_temporal_init).unwrap();
    ctx.stream.synchronize().unwrap();

    let mut grads = GpuMamba3Grads::new(&ctx.stream, cfg, dims.mamba_input_dim).unwrap();
    grads.zero(&ctx.stream).unwrap();
    ctx.stream.synchronize().unwrap();

    gpu_backward_mamba3_backbone(exec, &mut d_temporal, &acts, &w, &grads, &mut scratch).unwrap();
    ctx.stream.synchronize().unwrap();

    let mut dt_out = vec![0f32; bt * dims.d_model];
    d_temporal.download(&ctx.stream, &mut dt_out).unwrap();
    let mut arena = vec![0f32; grads.flat.len()];
    grads.flat.download(&ctx.stream, &mut arena).unwrap();
    (dt_out, arena)
}

fn run_mixed(
    exec: &M3Exec<'_>,
    cpu: &Mamba3Weights,
    cfg: &Mamba3Config,
    dtype: WeightDtype,
    mamba_input: &[f32],
    d_temporal_init: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let M3Exec { ctx, dims, .. } = *exec;
    let w =
        GpuMamba3TrainMixedWeights::from_cpu(&ctx.stream, cpu, cfg, dims.mamba_input_dim, dtype)
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
    mi.upload(&ctx.stream, mamba_input).unwrap();

    let mut temporal = GpuBuffer::zeros(&ctx.stream, bt * dims.d_model).unwrap();
    let mut acts = GpuMamba3BackboneMixedActs::new(
        &ctx.stream,
        cfg,
        b,
        dims.seq_len,
        dims.mamba_input_dim,
        dtype,
    )
    .unwrap();
    let mut mixed_scratch =
        GpuMamba3MixedScratch::new(&ctx.stream, cfg, b, dims.seq_len, dtype).unwrap();
    let mut f32_scratch = GpuMamba3Scratch::new(&ctx.stream, dims).unwrap();
    ctx.stream.synchronize().unwrap();

    gpu_forward_mamba3_backbone_mixed(
        exec,
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
        &mut mixed_scratch,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();

    let mut d_temporal = GpuBuffer::zeros(&ctx.stream, bt * dims.d_model).unwrap();
    ctx.stream.synchronize().unwrap();
    d_temporal.upload(&ctx.stream, d_temporal_init).unwrap();
    ctx.stream.synchronize().unwrap();

    let mut grads = GpuMamba3Grads::new(&ctx.stream, cfg, dims.mamba_input_dim).unwrap();
    grads.zero(&ctx.stream).unwrap();
    ctx.stream.synchronize().unwrap();

    gpu_backward_mamba3_backbone_mixed(
        exec,
        &mut d_temporal,
        &acts,
        &w,
        &grads,
        &mut f32_scratch,
        &mut mixed_scratch,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();

    let mut dt_out = vec![0f32; bt * dims.d_model];
    d_temporal.download(&ctx.stream, &mut dt_out).unwrap();
    let mut arena = vec![0f32; grads.flat.len()];
    grads.flat.download(&ctx.stream, &mut arena).unwrap();
    (dt_out, arena)
}

/// Per-tensor grad arena layout (mirrors GpuMamba3Grads::new in
/// src/mamba3_siso/gpu/weights.rs:367-403 so we can compare slice-by-slice).
fn grad_layout(cfg: &Mamba3Config, input_dim: usize) -> Vec<(&'static str, usize)> {
    let dm = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let nh = cfg.nheads();
    let ip = cfg.in_proj_out_dim();
    let mut out = Vec::new();
    out.push(("input_proj_w", input_dim * dm));
    out.push(("input_proj_b", dm));
    for _ in 0..cfg.n_layers {
        out.push(("norm_weight", dm));
        out.push(("in_proj_w", dm * ip));
        out.push(("dt_bias", nh));
        out.push(("b_norm_weight", ds));
        out.push(("c_norm_weight", ds));
        out.push(("b_bias", nh * ds));
        out.push(("c_bias", nh * ds));
        out.push(("d_param", nh));
        out.push(("norm_gate_weight", di));
        out.push(("out_proj_w", di * dm));
    }
    out.push(("norm_f_weight", dm));
    out
}

fn check(dtype: WeightDtype, t: usize) {
    check_cfg(cfg_for_step10(), dtype, t);
}

fn check_cfg(cfg: Mamba3Config, dtype: WeightDtype, t: usize) {
    let dims = dims_for(&cfg, 1, t);
    let (w_f32, w_mix) = build_weights(&cfg, 0xC0FFEE);

    let bt = dims.bt();
    let mamba_input = det_rand(bt * dims.mamba_input_dim, 0xB1);
    let d_temporal = det_rand(bt * dims.d_model, 0xB2);

    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = Mamba3Kernels::compile(dev.context(), "sm_89").unwrap();

    let exec = M3Exec {
        ctx: &ctx,
        kernels: &m3k,
        dims: &dims,
    };
    let (dt_ref, grads_ref) = run_f32(&exec, &w_f32, &cfg, &mamba_input, &d_temporal);
    let (dt_typ, grads_typ) = run_mixed(&exec, &w_mix, &cfg, dtype, &mamba_input, &d_temporal);

    assert_eq!(grads_ref.len(), grads_typ.len(), "grad arena sizes differ");

    let (cos_min, norm_tol) = match dtype {
        WeightDtype::Bf16 => (0.99_f32, 0.05_f32),
        WeightDtype::F16 => (0.999_f32, 0.02_f32),
        WeightDtype::F32 => unreachable!("f32 mixed path unsupported"),
    };

    eprintln!("m3_backward_mixed_parity {dtype:?} T={t}:");
    assert_close("d_temporal", &dt_ref, &dt_typ, cos_min, norm_tol);

    let layout = grad_layout(&cfg, dims.mamba_input_dim);
    let mut off = 0usize;
    for (label, len) in layout {
        // input_proj diverges by construction (eye vs identity-branch); skip.
        let skip = label == "input_proj_w" || label == "input_proj_b";
        if !skip {
            let r = &grads_ref[off..off + len];
            let t = &grads_typ[off..off + len];
            assert_close(label, r, t, cos_min, norm_tol);
        }
        off += len;
    }
    assert_eq!(off, grads_ref.len());
}

#[test]
fn m3_backward_mixed_parity_bf16() {
    check(WeightDtype::Bf16, 64);
}

#[test]
fn m3_backward_mixed_parity_f16() {
    check(WeightDtype::F16, 64);
}

/// T spans multiple chunks (chunk_size=64 in M3) — exercises the parallel
/// chunked bwd dispatch fully.
#[test]
fn m3_backward_mixed_parity_multi_chunk_bf16() {
    check(WeightDtype::Bf16, 192);
}

/// ngroups = nheads: per-head B/C (no group sharing). Exercises the
/// non-shared BCNorm path in the chunked bwd.
#[test]
fn m3_backward_mixed_parity_ngroups_eq_nheads_bf16() {
    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 8, // nheads = d_inner/headdim = 64/8 = 8 → per-head B/C
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    };
    check_cfg(cfg, WeightDtype::Bf16, 128);
}

/// Larger d_state (stresses SSM recurrence matmul sizes in backward).
#[test]
fn m3_backward_mixed_parity_large_d_state_bf16() {
    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 16,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    };
    check_cfg(cfg, WeightDtype::Bf16, 128);
}

/// Multi-layer config — exercises the per-layer state slice offsetting
/// and gradient accumulation across layers in backward_mixed.
#[test]
fn m3_backward_mixed_parity_two_layers_bf16() {
    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    };
    check_cfg(cfg, WeightDtype::Bf16, 128);
}
