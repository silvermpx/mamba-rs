//! The block tier of the Mamba-3 head kernels: bit-equal to the warp tier
//! at the widths both serve, and agreeing with the CPU at the widths only
//! the block tier serves, the official checkpoints' headdim 64 among them
//! and heads wider than a block.
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use cudarc::driver::PushKernelArg;
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
use mamba_rs::mamba3_siso::cpu::prefill::{Mamba3PrefillScratch, forward_mamba3_backbone_prefill};
use mamba_rs::mamba3_siso::gpu::inference::Mamba3GpuEngine;
use mamba_rs::mamba3_siso::gpu::kernels::{Mamba3Kernels, step_head_cfg};
use mamba_rs::mamba3_siso::route::HeadTier;
use mamba_rs::mamba3_siso::state::Mamba3State;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::mamba3_siso::{Mamba3StepScratch, mamba3_step};

fn det(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn upload_typed(ctx: &GpuCtx, data: &[f32], dtype: WeightDtype) -> DtypedBuf {
    let buf = DtypedBuf::zeros(&ctx.stream, data.len(), dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    buf.upload_f32(&ctx.stream, data).unwrap();
    ctx.stream.synchronize().unwrap();
    buf
}

fn download_typed(ctx: &GpuCtx, buf: &DtypedBuf) -> Vec<f32> {
    let mut out = vec![0f32; buf.len_elems()];
    buf.download_f32(&ctx.stream, &mut out).unwrap();
    ctx.stream.synchronize().unwrap();
    out
}

fn download(ctx: &GpuCtx, buf: &GpuBuffer) -> Vec<f32> {
    ctx.stream.synchronize().unwrap();
    let mut out = vec![0f32; buf.len()];
    buf.download(&ctx.stream, &mut out).unwrap();
    out
}

fn assert_bits(label: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{label}: length");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        assert_eq!(w.to_bits(), g.to_bits(), "{label}[{i}]: {w} vs {g}");
    }
}

fn assert_close(label: &str, got: &[f32], want: &[f32], rel_tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let (mut dot, mut na, mut nb, mut num) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in got.iter().zip(want) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
        num += f64::from(x - y) * f64::from(x - y);
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
    let rel = (num / nb.max(1e-30)).sqrt();
    assert!(
        cos > 0.9999 && rel < rel_tol,
        "{label}: cos={cos:.6} rel_l2={rel:.3e} (tol {rel_tol:.1e})"
    );
}

/// The step state of a (batch, heads, width, state) shape, one copy per
/// tier, so each tier mutates its own.
struct StepState {
    ssm: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
}

impl StepState {
    fn new(ctx: &GpuCtx, ssm: &[f32], k: &[f32], v: &[f32]) -> Self {
        let s = &ctx.stream;
        Self {
            ssm: GpuBuffer::from_cpu(s, ssm).unwrap(),
            k: GpuBuffer::from_cpu(s, k).unwrap(),
            v: GpuBuffer::from_cpu(s, v).unwrap(),
        }
    }
}

/// Both step kernels, launched directly on identical inputs for two steps
/// in a row (so the K and V carried from the first step feed the second),
/// must leave identical bits in the state and the output at every width
/// the warp tier serves, in every activation dtype.
#[test]
fn m3_step_block_matches_warp_bits() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k =
        Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), 64).unwrap();
    let (batch, nh) = (3usize, 2usize);
    for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
        for hd in [4usize, 8, 16, 32] {
            for ds in [8usize, 16, 64] {
                let label = format!("{dtype:?} hd={hd} ds={ds}");
                let di = nh * hd;
                let ssm0 = det(batch * nh * hd * ds, 11, 0.5);
                let k0 = det(batch * nh * ds, 12, 0.5);
                let v0 = det(batch * nh * hd, 13, 0.5);
                let d = GpuBuffer::from_cpu(&ctx.stream, &det(nh, 14, 1.0)).unwrap();
                let mut tiers = [
                    (HeadTier::Warp, StepState::new(&ctx, &ssm0, &k0, &v0)),
                    (HeadTier::Block, StepState::new(&ctx, &ssm0, &k0, &v0)),
                ];
                let mut ys: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
                for step in 0..2u64 {
                    let x = upload_typed(&ctx, &det(batch * di, 20 + step, 1.0), dtype);
                    let kc = upload_typed(&ctx, &det(batch * nh * ds, 30 + step, 1.0), dtype);
                    let qc = upload_typed(&ctx, &det(batch * nh * ds, 40 + step, 1.0), dtype);
                    let dt_vals: Vec<f32> = det(batch * nh, 50 + step, 1.0)
                        .into_iter()
                        .map(|v| 0.01 + v.abs() * 0.3)
                        .collect();
                    let a_vals: Vec<f32> = det(batch * nh, 60 + step, 1.0)
                        .into_iter()
                        .map(|v| -0.1 - v.abs())
                        .collect();
                    let trap_vals: Vec<f32> = det(batch * nh, 70 + step, 1.0)
                        .into_iter()
                        .map(|v| 0.5 + 0.4 * v)
                        .collect();
                    let dt = GpuBuffer::from_cpu(&ctx.stream, &dt_vals).unwrap();
                    let a = GpuBuffer::from_cpu(&ctx.stream, &a_vals).unwrap();
                    let trap = GpuBuffer::from_cpu(&ctx.stream, &trap_vals).unwrap();
                    for (slot, (tier, st)) in tiers.iter_mut().enumerate() {
                        let y = upload_typed(&ctx, &vec![f32::NAN; batch * di], dtype);
                        let kernel = match tier {
                            HeadTier::Warp => m3k.m3_step_fwd_typed.i32.get(dtype),
                            HeadTier::Block => m3k.m3_step_fwd_block_typed.i32.get(dtype),
                        };
                        let cfg =
                            step_head_cfg(*tier, batch, nh, hd, ds, m3k.step_block_threads.i32);
                        let ptrs = [
                            st.ssm.cached_ptr(),
                            st.k.cached_ptr(),
                            st.v.cached_ptr(),
                            y.cached_ptr(),
                            x.cached_ptr(),
                            kc.cached_ptr(),
                            qc.cached_ptr(),
                            dt.cached_ptr(),
                            a.cached_ptr(),
                            trap.cached_ptr(),
                            d.cached_ptr(),
                        ];
                        let dims = [batch, nh, hd, ds].map(|v| v as i32);
                        let mut bld = ctx.stream.launch_builder(kernel);
                        for ptr in &ptrs {
                            bld.arg(ptr);
                        }
                        for dim in &dims {
                            bld.arg(dim);
                        }
                        unsafe { bld.launch(cfg) }.unwrap();
                        ctx.stream.synchronize().unwrap();
                        ys[slot].push(download_typed(&ctx, &y));
                    }
                }
                let [(_, warp), (_, block)] = &tiers;
                for (name, want, got) in [
                    ("ssm_state", &warp.ssm, &block.ssm),
                    ("k_state", &warp.k, &block.k),
                    ("v_state", &warp.v, &block.v),
                ] {
                    assert_bits(
                        &format!("{label} {name}"),
                        &download(&ctx, want),
                        &download(&ctx, got),
                    );
                }
                for (step, (want, got)) in ys[0].iter().zip(&ys[1]).enumerate() {
                    assert_bits(&format!("{label} y step {step}"), want, got);
                }
            }
        }
    }
}

/// A small model at a width only the block tier serves, with a live input
/// projection: the CPU step always runs it. `gated` puts the gated output
/// norm in place of the SiLU gate.
fn block_cfg(
    d_model: usize,
    headdim: usize,
    d_state: usize,
    gated: bool,
) -> (Mamba3Config, Mamba3Weights) {
    let cfg = Mamba3Config {
        d_model,
        d_state,
        expand: 2,
        headdim,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 1e-4,
        is_outproj_norm: gated,
        ..Mamba3Config::default()
    };
    cfg.validate().unwrap();
    let w = Mamba3Weights::init(&cfg, d_model, 0xB10C);
    (cfg, w)
}

// CPU against GPU is a tolerance comparison: 2e-3 is the sm_120 floor the
// chunked prefill parity measured, with headroom, and a wrong broadcast
// fails it by orders of magnitude.
const REL_TOL: f64 = 2e-3;

/// Decode steps on the GPU engine at block-tier widths, headdim 64 with
/// d_state 128 among them, against the CPU step chain.
#[test]
fn m3_block_tier_decode_matches_cpu() {
    let device = GpuDevice::new(0).unwrap();
    for (d_model, hd, ds, gated) in [
        (128usize, 64usize, 128usize, false),
        (96, 48, 64, false),
        (128, 128, 32, false),
        (640, 64, 32, true),
        (96, 48, 16, true),
    ] {
        let (cfg, w) = block_cfg(d_model, hd, ds, gated);
        let engine = Mamba3GpuEngine::new(&device, &w, cfg, d_model, 1).unwrap();
        let mut gpu_state = engine.alloc_state().unwrap();
        let mut gpu_scratch = engine.alloc_scratch().unwrap();
        let mut cpu_state = Mamba3State::zeros(&cfg);
        let mut cpu_scratch = Mamba3StepScratch::new(&cfg);
        let mut cpu_out = vec![0f32; d_model];
        let mut gpu_out = vec![0f32; d_model];
        for t in 0..12u64 {
            let input = det(d_model, 100 + t, 0.5);
            mamba3_step(
                &mut cpu_out,
                &input,
                &mut cpu_scratch,
                &w,
                &mut cpu_state.layers,
                &cfg,
            );
            engine
                .step(&input, &mut gpu_out, &mut gpu_state, &mut gpu_scratch)
                .unwrap();
            assert_close(
                &format!("hd={hd} ds={ds} step {t}"),
                &gpu_out,
                &cpu_out,
                REL_TOL,
            );
        }
    }
}

/// A prompt window prefilled on the GPU, its length not a whole number of
/// chunks, then decoded step by step, against the CPU prefill and step
/// chain.
fn check_prefill_then_decode(device: &GpuDevice, cfg: Mamba3Config, w: &Mamba3Weights) {
    let d_model = cfg.d_model;
    let seq_len = 37usize;
    let prompt = det(seq_len * d_model, 7, 0.5);

    let dims = Mamba3Dims::from_config(&cfg, seq_len);
    let mut cpu_state = Mamba3State::zeros(&cfg);
    let mut cpu_prefill = Mamba3PrefillScratch::new(&dims, d_model);
    let mut cpu_rows = vec![0f32; seq_len * d_model];
    forward_mamba3_backbone_prefill(
        &mut cpu_rows,
        &prompt,
        w,
        &mut cpu_state.layers,
        &mut cpu_prefill,
        &dims,
    );

    let engine = Mamba3GpuEngine::new(device, w, cfg, d_model, 1).unwrap();
    let stream = &engine.ctx().stream;
    let mut prefill = engine.alloc_prefill(seq_len).unwrap();
    let mut gpu_state = engine.alloc_state().unwrap();
    let mut gpu_scratch = engine.alloc_scratch().unwrap();
    let gpu_prompt = GpuBuffer::from_cpu(stream, &prompt).unwrap();
    let mut last_hidden = GpuBuffer::zeros(stream, d_model).unwrap();
    engine
        .prefill_sequence(
            &mut prefill,
            &gpu_prompt,
            seq_len,
            &mut gpu_state,
            false,
            &mut last_hidden,
        )
        .unwrap();
    assert_close(
        "prefill last hidden",
        &download(engine.ctx(), &last_hidden),
        &cpu_rows[(seq_len - 1) * d_model..],
        REL_TOL,
    );

    let mut cpu_scratch = Mamba3StepScratch::new(&cfg);
    let mut cpu_out = vec![0f32; d_model];
    let mut gpu_out = vec![0f32; d_model];
    for t in 0..6u64 {
        let input = det(d_model, 300 + t, 0.5);
        mamba3_step(
            &mut cpu_out,
            &input,
            &mut cpu_scratch,
            w,
            &mut cpu_state.layers,
            &cfg,
        );
        engine
            .step(&input, &mut gpu_out, &mut gpu_state, &mut gpu_scratch)
            .unwrap();
        assert_close(
            &format!("decode after prefill, step {t}"),
            &gpu_out,
            &cpu_out,
            REL_TOL,
        );
    }
}

/// Headdim 64 with d_state 128, the official checkpoints' head.
#[test]
fn m3_block_tier_prefill_then_decode_matches_cpu() {
    let device = GpuDevice::new(0).unwrap();
    let (cfg, w) = block_cfg(128, 64, 128, false);
    check_prefill_then_decode(&device, cfg, &w);
}

/// The gated output norm at widths the one-block-per-row kernel never
/// served: 1280 channels in heads of 64, and heads of 48.
#[test]
fn m3_gated_norm_wide_and_ragged_prefill_matches_cpu() {
    let device = GpuDevice::new(0).unwrap();
    for (d_model, hd, ds) in [(640usize, 64usize, 32usize), (96, 48, 16)] {
        let (cfg, w) = block_cfg(d_model, hd, ds, true);
        check_prefill_then_decode(&device, cfg, &w);
    }
}

/// Heads wider than a block: the decode step, the chunked forward's
/// fallback tiles and the gated output norm take a head's elements in
/// turns. Heads of 640 and 1100, the second a gated norm group past 1024
/// channels.
#[test]
fn m3_heads_wider_than_a_block_prefill_then_decode_matches_cpu() {
    let device = GpuDevice::new(0).unwrap();
    for (d_model, hd, ds, gated) in [
        (320usize, 640usize, 32usize, true),
        (550, 1100, 16, false),
        (550, 1100, 16, true),
    ] {
        let (cfg, w) = block_cfg(d_model, hd, ds, gated);
        check_prefill_then_decode(&device, cfg, &w);
    }
}
