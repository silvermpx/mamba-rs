//! workload-style training benchmarks via the high-level
//! `MambaTrainer` / `Mamba3Trainer` API, sweeping all three precisions
//! (f32 / bf16 / f16) × eager vs CUDA Graph.
//!
//! Two workloads:
//!   * **LM** — Mamba-1 at mamba-130m-ish shape (d_model=768, n_layers=24,
//!     batch=2, seq_len=64) — full forward+backward+AdamW+sync per step
//!   * **RL** — Mamba-3 at SQV-RS actor shape (d_model=128, n_layers=4,
//!     headdim=16, ngroups=1, batch=64, seq_len=32)
//!
//! All tests are `#[ignore]` — opt-in:
//!   cargo test --release --features cuda --test trainer_benchmarks -- --ignored --nocapture
//!
//! The outputs are eager-vs-graph timing tables for each dtype, written
//! to stderr. Use these to track the real cost of a training step at
//! release-realistic shapes.

#![cfg(feature = "cuda")]

mod common;

use std::time::Instant;

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

const WARMUP: usize = 3;
const STEPS_EAGER: usize = 10;
const STEPS_GRAPH: usize = 30;

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

// ═══════════════════════════════════════════════════════════════════════
// LM workload — Mamba-1 at 130m-ish shape, all 3 dtypes
// ═══════════════════════════════════════════════════════════════════════

fn lm_cfg(scan_mode: mamba_rs::config::ScanMode) -> mamba_rs::config::MambaConfig {
    mamba_rs::config::MambaConfig {
        d_model: 768,
        n_layers: 24,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode,
        rms_norm_eps: 1e-5,
    }
}

fn run_lm_for_dtype(dtype: WeightDtype) -> Result<(), String> {
    run_lm(dtype, mamba_rs::config::ScanMode::Sequential, 64, "")
}

fn run_lm(
    dtype: WeightDtype,
    scan_mode: mamba_rs::config::ScanMode,
    seq_len: usize,
    suffix: &str,
) -> Result<(), String> {
    run_lm_shape(dtype, lm_cfg(scan_mode), 2, seq_len, suffix)
}

fn run_lm_shape(
    dtype: WeightDtype,
    cfg: mamba_rs::config::MambaConfig,
    batch: usize,
    seq_len: usize,
    suffix: &str,
) -> Result<(), String> {
    use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
    use mamba_rs::weights::MambaWeights;

    let input_dim = cfg.d_model;
    let n = batch * seq_len * input_dim;
    let label = format!("{dtype:?}{suffix}");

    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    if !matches!(dtype, WeightDtype::F32) {
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
    }
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }

    // Conservative lr so f16 doesn't NaN out before timing finishes.
    let mut trainer = MambaTrainer::new_full(
        0,
        &cpu,
        cfg,
        TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr: 1e-7,
            weight_decay: 0.0,
        },
        dtype,
    )?;
    // Honest-f32 row: mamba-rs f32 GEMMs run TF32 by handle default
    // while torch's "f32" is IEEE — opt into an IEEE row explicitly.
    if std::env::var("MAMBA_RS_BENCH_IEEE_F32").as_deref() == Ok("1") {
        trainer.ctx().disable_tf32();
        eprintln!("note: MAMBA_RS_BENCH_IEEE_F32=1 — f32 GEMMs run IEEE (TF32 off)");
    }

    // Inputs are pre-generated OUTSIDE every timed region: the previous
    // version ran a serial 98k-iteration host RNG + two Vec collects
    // INSIDE both timers, polluting the eager/graph ratio.
    let ring: Vec<(Vec<f32>, Vec<f32>)> = (0..4)
        .map(|s| (det(n, 0xC0 + s as u32, 0.01), det(n, 0xD0 + s as u32, 0.01)))
        .collect();

    // Warmup
    for s in 0..WARMUP {
        let (a, b) = &ring[s % ring.len()];
        trainer.step(a, b)?;
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;

    // Eager timing
    let t0 = Instant::now();
    for s in 0..STEPS_EAGER {
        let (a, b) = &ring[s % ring.len()];
        trainer.step(a, b)?;
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let eager_ms = t0.elapsed().as_secs_f64() * 1000.0 / STEPS_EAGER as f64;

    // Capture + graph timing
    trainer.capture_graph()?;
    assert!(trainer.has_graph());
    let t1 = Instant::now();
    for s in 0..STEPS_GRAPH {
        let (a, b) = &ring[s % ring.len()];
        let m = trainer.step(a, b)?;
        assert!(m.graph_replayed);
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let graph_ms = t1.elapsed().as_secs_f64() * 1000.0 / STEPS_GRAPH as f64;

    // Self-describing measurement: the GEMM tier rides TWO env flags
    // (MAMBA_RS_BATCH_INVARIANT and MAMBA_RS_BI_TENSOR_CORES on top of
    // it); printing the resolved tier makes a missing base flag visible
    // in the reading itself.
    let tier = if trainer.ctx().batch_invariant() {
        if trainer.ctx().bi_tensor_cores() {
            "bi+tc"
        } else {
            "bi"
        }
    } else {
        "cublas"
    };
    eprintln!(
        "LM {label:5} tier={tier}  eager={eager_ms:7.3} ms/step  graph={graph_ms:7.3} ms/step  speedup={:.2}x",
        eager_ms / graph_ms
    );
    Ok(())
}

#[test]
#[ignore]
fn bench_lm_train_f32() {
    run_lm_for_dtype(WeightDtype::F32).unwrap();
}

#[test]
#[ignore]
fn bench_lm_train_bf16() {
    run_lm_for_dtype(WeightDtype::Bf16).unwrap();
}

#[test]
#[ignore]
fn bench_lm_train_f16() {
    run_lm_for_dtype(WeightDtype::F16).unwrap();
}

/// The parallel scan has NEVER been measured at this shape (Auto
/// resolves Sequential below T=256) — first-ever A/B arm.
#[test]
#[ignore]
fn bench_lm_train_f32_parallel_scan() {
    run_lm(
        WeightDtype::F32,
        mamba_rs::config::ScanMode::Parallel,
        64,
        " par",
    )
    .unwrap();
}

#[test]
#[ignore]
fn bench_lm_train_bf16_parallel_scan() {
    run_lm(
        WeightDtype::Bf16,
        mamba_rs::config::ScanMode::Parallel,
        64,
        " par",
    )
    .unwrap();
}

/// Free-shape bench arm for measuring the ACTUAL training shape instead
/// of extrapolating from the B2xT64 microbench (the P0.4 honesty rule:
/// the campaign wall is measured at the campaign shape). Every knob
/// rides an env var so no recompile is needed per shape:
///   MAMBA_RS_BENCH_DM (d_model, default 384)
///   MAMBA_RS_BENCH_LAYERS (default 24)
///   MAMBA_RS_BENCH_B (batch, default 8)
///   MAMBA_RS_BENCH_T (seq_len, default 1300)
///   MAMBA_RS_BENCH_DTYPE (f32|bf16|f16, default bf16)
///   MAMBA_RS_BENCH_SCAN (auto|seq|par, default auto)
/// Combine with MAMBA_RS_BATCH_INVARIANT=1 (+_TC=1) for the BI tiers.
#[test]
#[ignore]
fn bench_lm_train_campaign_shape() {
    let get = |k: &str, d: usize| -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let dm = get("MAMBA_RS_BENCH_DM", 384);
    let layers = get("MAMBA_RS_BENCH_LAYERS", 24);
    let b = get("MAMBA_RS_BENCH_B", 8);
    let t = get("MAMBA_RS_BENCH_T", 1300);
    let dtype = match std::env::var("MAMBA_RS_BENCH_DTYPE").as_deref() {
        Ok("f32") => WeightDtype::F32,
        Ok("f16") => WeightDtype::F16,
        _ => WeightDtype::Bf16,
    };
    let scan = match std::env::var("MAMBA_RS_BENCH_SCAN").as_deref() {
        Ok("seq") => mamba_rs::config::ScanMode::Sequential,
        Ok("par") => mamba_rs::config::ScanMode::Parallel,
        _ => mamba_rs::config::ScanMode::Auto,
    };
    let cfg = mamba_rs::config::MambaConfig {
        d_model: dm,
        n_layers: layers,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: scan,
        rms_norm_eps: 1e-5,
    };
    eprintln!("campaign shape: dm={dm} L={layers} B={b} T={t} {dtype:?} scan={scan:?}");
    run_lm_shape(dtype, cfg, b, t, " campaign").unwrap();
}

/// Campaign-shape attribution: time forward() alone vs forward()+
/// backward_step() pairs through the split API (always eager) and
/// report the subtraction. Same env knobs as
/// `bench_lm_train_campaign_shape`.
#[test]
#[ignore]
fn bench_campaign_split_fwd_bwd() {
    use mamba_rs::mamba_ssm::gpu::trainer::{BackwardOpts, MambaTrainer, TrainSessionCfg};
    use mamba_rs::weights::MambaWeights;
    let get = |k: &str, d: usize| -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let dm = get("MAMBA_RS_BENCH_DM", 384);
    let layers = get("MAMBA_RS_BENCH_LAYERS", 24);
    let b = get("MAMBA_RS_BENCH_B", 8);
    let t = get("MAMBA_RS_BENCH_T", 1300);
    let dtype = match std::env::var("MAMBA_RS_BENCH_DTYPE").as_deref() {
        Ok("f32") => WeightDtype::F32,
        Ok("f16") => WeightDtype::F16,
        _ => WeightDtype::Bf16,
    };
    let cfg = mamba_rs::config::MambaConfig {
        d_model: dm,
        n_layers: layers,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Auto,
        rms_norm_eps: 1e-5,
    };
    let input_dim = dm;
    let n = b * t * dm;
    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    if !matches!(dtype, WeightDtype::F32) {
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
    }
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    let mut trainer = MambaTrainer::new_full(
        0,
        &cpu,
        cfg,
        TrainSessionCfg {
            input_dim,
            batch: b,
            seq_len: t,
            lr: 1e-7,
            weight_decay: 0.0,
        },
        dtype,
    )
    .unwrap();
    let inp = det(n, 0xA1, 0.01);
    let dt = det(n, 0xB1, 0.01);
    let mut out = vec![0f32; n];
    for _ in 0..3 {
        trainer.forward(&inp, &mut out).unwrap();
        trainer.backward_step(&dt, BackwardOpts::default()).unwrap();
    }
    trainer.ctx().stream.synchronize().unwrap();
    let reps = 10;
    let t0 = Instant::now();
    for _ in 0..reps {
        trainer.forward(&inp, &mut out).unwrap();
    }
    trainer.ctx().stream.synchronize().unwrap();
    let fwd_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
    // Leave no dangling saved-activation state: pair timing next.
    trainer.backward_step(&dt, BackwardOpts::default()).unwrap();
    trainer.ctx().stream.synchronize().unwrap();
    let t1 = Instant::now();
    for _ in 0..reps {
        trainer.forward(&inp, &mut out).unwrap();
        trainer.backward_step(&dt, BackwardOpts::default()).unwrap();
    }
    trainer.ctx().stream.synchronize().unwrap();
    let pair_ms = t1.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
    eprintln!(
        "campaign split dm={dm} L={layers} B={b} T={t} {dtype:?}: fwd={fwd_ms:.1} ms  bwd+opt={:.1} ms  pair={pair_ms:.1} ms",
        pair_ms - fwd_ms
    );
}

/// One point OFF the batch-invariant `batch >= 128` dispatch boundary
/// (B2 x T64 lands exactly ON it): run under MAMBA_RS_BATCH_INVARIANT=1
/// to see the small-M bucket family.
#[test]
#[ignore]
fn bench_lm_train_f32_t60() {
    run_lm(
        WeightDtype::F32,
        mamba_rs::config::ScanMode::Sequential,
        60,
        " t60",
    )
    .unwrap();
}

// ═══════════════════════════════════════════════════════════════════════
// RL workload — Mamba-3 actor shape (SQV-RS-ish), all 3 dtypes
// ═══════════════════════════════════════════════════════════════════════

fn rl_cfg() -> mamba_rs::mamba3_siso::config::Mamba3Config {
    // Note: `is_outproj_norm: true` to enable the M3 mixed backward path
    // (the no-norm gating variant isn't wired for mixed bwd). For pure
    // f32 training either flag works.
    mamba_rs::mamba3_siso::config::Mamba3Config {
        d_model: 128,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 4,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
        ..mamba_rs::mamba3_siso::config::Mamba3Config::default()
    }
}

fn run_rl_for_dtype(dtype: WeightDtype) -> Result<(), String> {
    use mamba_rs::mamba3_siso::gpu::trainer::{Mamba3Trainer, TrainSessionCfg};
    use mamba_rs::mamba3_siso::weights::Mamba3Weights;

    let cfg = rl_cfg();
    let input_dim = cfg.d_model;
    let batch = 64;
    let seq_len = 32;
    let n = batch * seq_len * input_dim;
    let label = format!("{dtype:?}");

    let mut cpu = Mamba3Weights::init(&cfg, input_dim, 0xDECADE);
    if matches!(dtype, WeightDtype::F32) {
        // f32 M3 forward needs a real input_proj (no identity branch).
        cpu.input_proj_w = (0..input_dim * cfg.d_model)
            .map(|i| {
                if i / cfg.d_model == i % cfg.d_model {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        cpu.input_proj_b = vec![0.0; cfg.d_model];
    } else {
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
    }

    let mut trainer = Mamba3Trainer::new_full(
        0,
        &cpu,
        cfg,
        TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr: 1e-7,
            weight_decay: 0.0,
        },
        dtype,
    )?;

    for s in 0..WARMUP {
        trainer.step(
            &det(n, 0xA0 + s as u32, 0.01),
            &det(n, 0xB0 + s as u32, 0.01),
        )?;
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;

    let t0 = Instant::now();
    for s in 0..STEPS_EAGER {
        trainer.step(
            &det(n, 0xC0 + s as u32, 0.01),
            &det(n, 0xD0 + s as u32, 0.01),
        )?;
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let eager_ms = t0.elapsed().as_secs_f64() * 1000.0 / STEPS_EAGER as f64;

    trainer.capture_graph()?;
    let t1 = Instant::now();
    for s in 0..STEPS_GRAPH {
        let m = trainer.step(
            &det(n, 0xE0 + s as u32, 0.01),
            &det(n, 0xF0 + s as u32, 0.01),
        )?;
        assert!(m.graph_replayed);
    }
    trainer
        .ctx()
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let graph_ms = t1.elapsed().as_secs_f64() * 1000.0 / STEPS_GRAPH as f64;

    eprintln!(
        "RL {label:5}  eager={eager_ms:7.3} ms/step  graph={graph_ms:7.3} ms/step  speedup={:.2}x",
        eager_ms / graph_ms
    );
    Ok(())
}

#[test]
#[ignore]
fn bench_rl_train_f32() {
    run_rl_for_dtype(WeightDtype::F32).unwrap();
}

#[test]
#[ignore]
fn bench_rl_train_bf16() {
    run_rl_for_dtype(WeightDtype::Bf16).unwrap();
}

#[test]
#[ignore]
fn bench_rl_train_f16() {
    run_rl_for_dtype(WeightDtype::F16).unwrap();
}

/// S0 instrument: time the parallel-scan kernels IN ISOLATION at the
/// campaign shape (one layer). Attributes the campaign wall at kernel
/// granularity — every prior fix that "should" have moved the 441 ms
/// step left it untouched, so nothing else gets built before this
/// number exists.
#[test]
#[ignore]
fn bench_scan_kernels_isolated() {
    use cudarc::driver::PushKernelArg;
    use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
    use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::launch::{grid_parallel_scan_bwd, grid_parallel_scan_typed};

    let (b, t, di, ds) = (8usize, 1300usize, 768usize, 16usize);
    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new_with_state_cap(&device, 16).unwrap();
    let k = &ctx.kernels;
    let dtype = WeightDtype::Bf16;

    let bt = b * t;
    let upload_typed = |data: &[f32]| -> DtypedBuf {
        let buf = DtypedBuf::zeros(&ctx.stream, data.len(), dtype).unwrap();
        ctx.stream.synchronize().unwrap();
        buf.upload_f32(&ctx.stream, data).unwrap();
        ctx.stream.synchronize().unwrap();
        buf
    };
    let upload_f32 = |data: &[f32]| -> GpuBuffer {
        let mut buf = GpuBuffer::zeros(&ctx.stream, data.len()).unwrap();
        ctx.stream.synchronize().unwrap();
        buf.upload(&ctx.stream, data).unwrap();
        ctx.stream.synchronize().unwrap();
        buf
    };
    let h = GpuBuffer::zeros(&ctx.stream, b * di * ds).unwrap();
    let y = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let delta = upload_typed(&det(bt * di, 11, 0.05));
    let u = upload_typed(&det(bt * di, 12, 0.5));
    let bb = upload_typed(&det(bt * ds, 13, 0.3));
    let cc = upload_typed(&det(bt * ds, 14, 0.3));
    let a_neg = upload_f32(
        &det(di * ds, 15, 0.2)
            .iter()
            .map(|x| -x.abs())
            .collect::<Vec<_>>(),
    );
    let dpar = upload_f32(&det(di, 16, 0.1));
    let h_saved = GpuBuffer::zeros(&ctx.stream, b * (t + 1) * di * ds).unwrap();

    let bi = b as i32;
    let ti = t as i32;
    let dii = di as i32;
    let dsi = ds as i32;

    let fwd = |ctx: &GpuCtx| {
        let mut bld = ctx
            .stream
            .launch_builder(k.ssm_parallel_fwd_typed.get(dtype));
        let hp = h.cached_ptr();
        let yp = y.cached_ptr();
        let dp = delta.cached_ptr();
        let up = u.cached_ptr();
        let bp = bb.cached_ptr();
        let cp = cc.cached_ptr();
        let ap = a_neg.cached_ptr();
        let ddp = dpar.cached_ptr();
        let hs = h_saved.cached_ptr();
        // Arg order matches the kernel signature (h, y, h_saved, delta,
        // u, B, C, a_neg, D, ...): the arm previously pushed h_saved
        // ninth, which shifted every pointer after y by one slot.
        let slim0: i32 = 0;
        bld.arg(&hp);
        bld.arg(&yp);
        bld.arg(&hs);
        bld.arg(&dp);
        bld.arg(&up);
        bld.arg(&bp);
        bld.arg(&cp);
        bld.arg(&ap);
        bld.arg(&ddp);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&dii);
        bld.arg(&dsi);
        bld.arg(&hs);
        bld.arg(&slim0);
        unsafe { bld.launch(grid_parallel_scan_typed(b, di, 2, ds)) }.unwrap();
    };
    let fwd_tape = GpuBuffer::zeros(
        &ctx.stream,
        mamba_rs::mamba_ssm::gpu::launch::scan_tape_len(b, t, di, ds),
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let fwd_slim = |ctx: &GpuCtx| {
        let mut bld = ctx
            .stream
            .launch_builder(k.ssm_parallel_fwd_typed.get(dtype));
        let hp = h.cached_ptr();
        let yp = y.cached_ptr();
        let dp = delta.cached_ptr();
        let up = u.cached_ptr();
        let bp = bb.cached_ptr();
        let cp = cc.cached_ptr();
        let ap = a_neg.cached_ptr();
        let ddp = dpar.cached_ptr();
        let hs = h_saved.cached_ptr();
        let tp = fwd_tape.cached_ptr();
        let slim1: i32 = 1;
        bld.arg(&hp);
        bld.arg(&yp);
        bld.arg(&hs);
        bld.arg(&dp);
        bld.arg(&up);
        bld.arg(&bp);
        bld.arg(&cp);
        bld.arg(&ap);
        bld.arg(&ddp);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&dii);
        bld.arg(&dsi);
        bld.arg(&tp);
        bld.arg(&slim1);
        unsafe { bld.launch(grid_parallel_scan_typed(b, di, 2, ds)) }.unwrap();
    };

    for _ in 0..3 {
        fwd(&ctx);
    }
    ctx.stream.synchronize().unwrap();
    let reps: i32 = 20;
    let t0 = Instant::now();
    for _ in 0..reps {
        fwd(&ctx);
    }
    ctx.stream.synchronize().unwrap();
    let fwd_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
    for _ in 0..3 {
        fwd_slim(&ctx);
    }
    ctx.stream.synchronize().unwrap();
    let t0s = Instant::now();
    for _ in 0..reps {
        fwd_slim(&ctx);
    }
    ctx.stream.synchronize().unwrap();
    let fwd_slim_ms = t0s.elapsed().as_secs_f64() * 1e3 / f64::from(reps);

    // bwd
    let d_y = upload_typed(&det(bt * di, 17, 0.1));
    let d_delta = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let d_u = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let d_b_local = DtypedBuf::zeros(&ctx.stream, bt * di * ds, dtype).unwrap();
    let d_c_local = DtypedBuf::zeros(&ctx.stream, bt * di * ds, dtype).unwrap();
    let d_d_local = GpuBuffer::zeros(&ctx.stream, b * di).unwrap();
    let d_a_log_local = GpuBuffer::zeros(&ctx.stream, b * di * ds).unwrap();

    let bwd = |ctx: &GpuCtx| {
        let mut bld = ctx
            .stream
            .launch_builder(k.ssm_parallel_bwd_typed.get(dtype));
        let hs = h_saved.cached_ptr();
        let dp = delta.cached_ptr();
        let up = u.cached_ptr();
        let bp = bb.cached_ptr();
        let cp = cc.cached_ptr();
        let ap = a_neg.cached_ptr();
        let ddp = dpar.cached_ptr();
        let dyp = d_y.cached_ptr();
        let ddel = d_delta.cached_ptr();
        let dup = d_u.cached_ptr();
        let dbl = d_b_local.cached_ptr();
        let dcl = d_c_local.cached_ptr();
        let ddl = d_d_local.cached_ptr();
        let dal = d_a_log_local.cached_ptr();
        bld.arg(&hs);
        bld.arg(&dp);
        bld.arg(&up);
        bld.arg(&bp);
        bld.arg(&cp);
        bld.arg(&ap);
        bld.arg(&ddp);
        bld.arg(&dyp);
        bld.arg(&ddel);
        bld.arg(&dup);
        bld.arg(&dbl);
        bld.arg(&dcl);
        bld.arg(&ddl);
        bld.arg(&dal);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&dii);
        bld.arg(&dsi);
        let slim0: i32 = 0;
        bld.arg(&hs);
        bld.arg(&slim0);
        unsafe { bld.launch(grid_parallel_scan_bwd(b, di)) }.unwrap();
    };
    // Production runs the slim tape (in-kernel h replay); time that
    // variant too so the ledger reflects the real trainer path.
    let tape = GpuBuffer::zeros(
        &ctx.stream,
        mamba_rs::mamba_ssm::gpu::launch::scan_tape_len(b, t, di, ds),
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();
    let bwd_slim = |ctx: &GpuCtx| {
        let mut bld = ctx
            .stream
            .launch_builder(k.ssm_parallel_bwd_typed.get(dtype));
        let hs = h_saved.cached_ptr();
        let tp = tape.cached_ptr();
        let dp = delta.cached_ptr();
        let up = u.cached_ptr();
        let bp = bb.cached_ptr();
        let cp = cc.cached_ptr();
        let ap = a_neg.cached_ptr();
        let ddp = dpar.cached_ptr();
        let dyp = d_y.cached_ptr();
        let ddel = d_delta.cached_ptr();
        let dup = d_u.cached_ptr();
        let dbl = d_b_local.cached_ptr();
        let dcl = d_c_local.cached_ptr();
        let ddl = d_d_local.cached_ptr();
        let dal = d_a_log_local.cached_ptr();
        bld.arg(&hs);
        bld.arg(&dp);
        bld.arg(&up);
        bld.arg(&bp);
        bld.arg(&cp);
        bld.arg(&ap);
        bld.arg(&ddp);
        bld.arg(&dyp);
        bld.arg(&ddel);
        bld.arg(&dup);
        bld.arg(&dbl);
        bld.arg(&dcl);
        bld.arg(&ddl);
        bld.arg(&dal);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&dii);
        bld.arg(&dsi);
        let slim1: i32 = 1;
        bld.arg(&tp);
        bld.arg(&slim1);
        unsafe { bld.launch(grid_parallel_scan_bwd(b, di)) }.unwrap();
    };
    for _ in 0..3 {
        bwd(&ctx);
    }
    ctx.stream.synchronize().unwrap();
    let t1 = Instant::now();
    for _ in 0..reps {
        bwd(&ctx);
    }
    ctx.stream.synchronize().unwrap();
    let bwd_ms = t1.elapsed().as_secs_f64() * 1e3 / f64::from(reps);

    for _ in 0..3 {
        bwd_slim(&ctx);
    }
    ctx.stream.synchronize().unwrap();
    let t2 = Instant::now();
    for _ in 0..reps {
        bwd_slim(&ctx);
    }
    ctx.stream.synchronize().unwrap();
    let bwd_slim_ms = t2.elapsed().as_secs_f64() * 1e3 / f64::from(reps);

    // The PRODUCTION route: fold kernel (dB/dC folded to di/G rows) with
    // the slim tape. The pre-0.6.4 ledger timed only the ungrouped
    // kernel, which the trainer launches solely when d_inner is not
    // divisible by the d-group - never at campaign shapes.
    use mamba_rs::mamba_ssm::gpu::launch::{SCAN_BWD_DGROUP, grid_parallel_scan_bwd_fold};
    let d_b_fold = DtypedBuf::zeros(&ctx.stream, bt * (di / SCAN_BWD_DGROUP) * ds, dtype).unwrap();
    let d_c_fold = DtypedBuf::zeros(&ctx.stream, bt * (di / SCAN_BWD_DGROUP) * ds, dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    let bwd_fold_slim = |ctx: &GpuCtx| {
        let mut bld = ctx
            .stream
            .launch_builder(k.ssm_parallel_bwd_fold_typed.get(dtype));
        let hs = h_saved.cached_ptr();
        let tp = tape.cached_ptr();
        let dp = delta.cached_ptr();
        let up = u.cached_ptr();
        let bp = bb.cached_ptr();
        let cp = cc.cached_ptr();
        let ap = a_neg.cached_ptr();
        let ddp = dpar.cached_ptr();
        let dyp = d_y.cached_ptr();
        let ddel = d_delta.cached_ptr();
        let dup = d_u.cached_ptr();
        let dbl = d_b_fold.cached_ptr();
        let dcl = d_c_fold.cached_ptr();
        let ddl = d_d_local.cached_ptr();
        let dal = d_a_log_local.cached_ptr();
        bld.arg(&hs);
        bld.arg(&dp);
        bld.arg(&up);
        bld.arg(&bp);
        bld.arg(&cp);
        bld.arg(&ap);
        bld.arg(&ddp);
        bld.arg(&dyp);
        bld.arg(&ddel);
        bld.arg(&dup);
        bld.arg(&dbl);
        bld.arg(&dcl);
        bld.arg(&ddl);
        bld.arg(&dal);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&dii);
        bld.arg(&dsi);
        let slim1: i32 = 1;
        bld.arg(&tp);
        bld.arg(&slim1);
        unsafe { bld.launch(grid_parallel_scan_bwd_fold(b, di, ds, dtype.size_bytes())) }.unwrap();
    };
    let bwd_fold_ms = common::bench::timed(&ctx, 20, || bwd_fold_slim(&ctx));

    eprintln!(
        "{}",
        common::bench::bench_stamp(
            &device,
            &ctx,
            "B8 T1300 di768 ds16",
            "fold G=4 slim",
            di / SCAN_BWD_DGROUP
        )
    );
    eprintln!(
        "scan isolated (B{b} T{t} di{di} ds{ds} {dtype:?}): fwd_full={fwd_ms:.3} (x24={:.1}) fwd_slim={fwd_slim_ms:.3} (x24={:.1}) bwd_full={bwd_ms:.3} (x24={:.1}) bwd_slim={bwd_slim_ms:.3} (x24={:.1}) bwd_fold_slim={bwd_fold_ms:.3} (x24={:.1}, production)",
        fwd_ms * 24.0,
        fwd_slim_ms * 24.0,
        bwd_ms * 24.0,
        bwd_slim_ms * 24.0,
        bwd_fold_ms * 24.0
    );
}

/// S0 part 2: the rest of the backward's suspects, isolated at the
/// campaign shape (one layer each).
#[test]
#[ignore]
fn bench_bwd_kernels_isolated() {
    use cudarc::driver::PushKernelArg;
    use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
    use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::launch::{grid_1d, grid_conv_tiled};

    let (b, t, di, ds, dc) = (8usize, 1300usize, 768usize, 16usize, 4usize);
    let bt = b * t;
    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new_with_state_cap(&device, 16).unwrap();
    let k = &ctx.kernels;
    let dtype = WeightDtype::Bf16;
    let bi = b as i32;
    let ti = t as i32;
    let dii = di as i32;
    let dsi = ds as i32;
    let dci = dc as i32;
    let bti = bt as i32;

    let d_u = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let post_conv = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let x_branch = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let conv_init = GpuBuffer::zeros(&ctx.stream, b * di * dc).unwrap();
    let weight = GpuBuffer::zeros(&ctx.stream, di * dc).unwrap();
    let d_x_branch = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let n_tiles = t.div_ceil(128);
    let wp = GpuBuffer::zeros(&ctx.stream, b * n_tiles * di * dc).unwrap();
    let bp = GpuBuffer::zeros(&ctx.stream, b * n_tiles * di).unwrap();

    let time_it = |label: &str, f: &dyn Fn()| {
        for _ in 0..3 {
            f();
        }
        ctx.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..20 {
            f();
        }
        ctx.stream.synchronize().unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3 / 20.0;
        eprintln!(
            "isolated {label}: {ms:.3} ms/layer (x24 = {:.1} ms)",
            ms * 24.0
        );
    };

    time_it("conv_dw_tiled", &|| {
        let mut bld = ctx
            .stream
            .launch_builder(k.conv1d_bwd_dw_tiled_typed.get(dtype));
        let wpp = wp.cached_ptr();
        let bpp = bp.cached_ptr();
        let dup = d_u.cached_ptr();
        let pcp = post_conv.cached_ptr();
        let xbp = x_branch.cached_ptr();
        let cip = conv_init.cached_ptr();
        bld.arg(&wpp);
        bld.arg(&bpp);
        bld.arg(&dup);
        bld.arg(&pcp);
        bld.arg(&xbp);
        bld.arg(&cip);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&dii);
        bld.arg(&dci);
        let lanes = b * di * (dc + 1);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (lanes.div_ceil(256) as u32, n_tiles as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { bld.launch(cfg) }.unwrap();
    });

    time_it("conv_dx_tiled", &|| {
        let mut bld = ctx
            .stream
            .launch_builder(k.conv1d_bwd_dx_tiled_typed.get(dtype));
        let dxp = d_x_branch.cached_ptr();
        let dup = d_u.cached_ptr();
        let pcp = post_conv.cached_ptr();
        let wpt = weight.cached_ptr();
        bld.arg(&dxp);
        bld.arg(&dup);
        bld.arg(&pcp);
        bld.arg(&wpt);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&dii);
        bld.arg(&dci);
        // Contiguous destination for the isolated arm (production passes
        // the d_proj geometry: stride 2*d_inner, offset 0).
        let stride_i = dii;
        let off_i = 0i32;
        bld.arg(&stride_i);
        bld.arg(&off_i);
        unsafe { bld.launch(grid_conv_tiled(b, di, t)) }.unwrap();
    });

    // gating backward
    let d_gated = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let d_y = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let d_gate = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let yb = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let gp = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    time_it("gating_bwd", &|| {
        let n = (bt * di) as i32;
        let mut bld = ctx.stream.launch_builder(k.gating_bwd_typed.get(dtype));
        let a1 = d_y.cached_ptr();
        let a2 = d_gate.cached_ptr();
        let a3 = d_gated.cached_ptr();
        let a4 = yb.cached_ptr();
        let a5 = gp.cached_ptr();
        bld.arg(&a1);
        bld.arg(&a2);
        bld.arg(&a3);
        bld.arg(&a4);
        bld.arg(&a5);
        bld.arg(&n);
        // Contiguous destination for the isolated arm (production passes
        // the d_proj geometry: stride 2*d_inner, offset d_inner).
        let dii2 = (bt * di) as i32;
        let off_i = 0i32;
        bld.arg(&dii2);
        bld.arg(&dii2);
        bld.arg(&off_i);
        unsafe { bld.launch(grid_1d(bt * di)) }.unwrap();
    });

    // fused dB/dC reducer (tmajor) - BOTH depths: production reduces the
    // FOLD output (di/G rows); the full-depth arm stays as the labelled
    // legacy reference (the pre-0.6.4 ledger row was this one only).
    use mamba_rs::mamba_ssm::gpu::launch::SCAN_BWD_DGROUP;
    let reduce_di = (di / SCAN_BWD_DGROUP) as i32;
    let d_b_local = DtypedBuf::zeros(&ctx.stream, bt * di * ds, dtype).unwrap();
    let d_c_local = DtypedBuf::zeros(&ctx.stream, bt * di * ds, dtype).unwrap();
    let d_b_fold = DtypedBuf::zeros(&ctx.stream, bt * (di / SCAN_BWD_DGROUP) * ds, dtype).unwrap();
    let d_c_fold = DtypedBuf::zeros(&ctx.stream, bt * (di / SCAN_BWD_DGROUP) * ds, dtype).unwrap();
    let d_b_red = GpuBuffer::zeros(&ctx.stream, bt * ds).unwrap();
    let d_c_red = GpuBuffer::zeros(&ctx.stream, bt * ds).unwrap();
    time_it("reduce_d_BC_tmajor depth=192 (production)", &|| {
        let mut bld = ctx
            .stream
            .launch_builder(k.ssm_reduce_d_bc_tmajor_typed.get(dtype));
        let o1 = d_b_red.cached_ptr();
        let o2 = d_c_red.cached_ptr();
        let i1 = d_b_fold.cached_ptr();
        let i2 = d_c_fold.cached_ptr();
        bld.arg(&o1);
        bld.arg(&o2);
        bld.arg(&i1);
        bld.arg(&i2);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&reduce_di);
        bld.arg(&dsi);
        unsafe { bld.launch(grid_1d(bt * ds)) }.unwrap();
    });
    time_it("reduce_d_BC_tmajor depth=768 (legacy)", &|| {
        let mut bld = ctx
            .stream
            .launch_builder(k.ssm_reduce_d_bc_tmajor_typed.get(dtype));
        let o1 = d_b_red.cached_ptr();
        let o2 = d_c_red.cached_ptr();
        let i1 = d_b_local.cached_ptr();
        let i2 = d_c_local.cached_ptr();
        bld.arg(&o1);
        bld.arg(&o2);
        bld.arg(&i1);
        bld.arg(&i2);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&dii);
        bld.arg(&dsi);
        unsafe { bld.launch(grid_1d(bt * ds)) }.unwrap();
    });
    let _ = bti;
}

/// S0 part 3: the four backward GEMM classes at exact campaign shapes,
/// through the SAME typed BI wrappers the trainer uses (TC tier on/off
/// via MAMBA_RS_BI_TENSOR_CORES).
#[test]
#[ignore]
fn bench_bwd_gemms_isolated() {
    use mamba_rs::mamba_ssm::gpu::blas::{
        TypedPtr, bi_sgemm_backward_dw_typed, bi_sgemm_backward_dx_typed,
    };
    use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
    use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;

    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new_with_state_cap(&device, 16).unwrap();
    ctx.set_batch_invariant(true);
    let dtype = WeightDtype::Bf16;
    let bt = 8usize * 1300;

    // (label, n_in, n_out) with batch = bt for each layer GEMM.
    let shapes = [
        ("in_proj", 384usize, 1536usize),
        ("x_proj", 768, 56),
        ("dt_proj", 24, 768),
        ("out_proj", 768, 384),
    ];
    for (label, n_in, n_out) in shapes {
        let dy = DtypedBuf::zeros(&ctx.stream, bt * n_out, dtype).unwrap();
        let x = DtypedBuf::zeros(&ctx.stream, bt * n_in, dtype).unwrap();
        let w = DtypedBuf::zeros(&ctx.stream, n_in * n_out, dtype).unwrap();
        let dx = DtypedBuf::zeros(&ctx.stream, bt * n_in, dtype).unwrap();
        let dw = GpuBuffer::zeros(&ctx.stream, n_in * n_out).unwrap();
        let run = |ctx: &GpuCtx| {
            bi_sgemm_backward_dw_typed(
                ctx,
                dw.cached_ptr(),
                TypedPtr {
                    ptr: dy.cached_ptr(),
                    dtype,
                },
                TypedPtr {
                    ptr: x.cached_ptr(),
                    dtype,
                },
                (bt, n_in, n_out),
            )
            .unwrap();
            bi_sgemm_backward_dx_typed(
                ctx,
                TypedPtr {
                    ptr: dx.cached_ptr(),
                    dtype,
                },
                TypedPtr {
                    ptr: dy.cached_ptr(),
                    dtype,
                },
                TypedPtr {
                    ptr: w.cached_ptr(),
                    dtype,
                },
                (bt, n_in, n_out),
            )
            .unwrap();
        };
        let run_dw = |ctx: &GpuCtx| {
            bi_sgemm_backward_dw_typed(
                ctx,
                dw.cached_ptr(),
                TypedPtr {
                    ptr: dy.cached_ptr(),
                    dtype,
                },
                TypedPtr {
                    ptr: x.cached_ptr(),
                    dtype,
                },
                (bt, n_in, n_out),
            )
            .unwrap();
        };
        // Tiny-kernel timing sits inside the GPU clock ramp with short
        // windows: 3+20 iterations read up to 7x slower cold than warm.
        // A ~1 s warm spin plus 200 timed iterations keeps readings
        // stable across cold and warm invocations.
        let time_one = |f: &dyn Fn(&GpuCtx)| {
            let warm = Instant::now();
            while warm.elapsed().as_secs_f64() < 1.0 {
                f(&ctx);
                ctx.stream.synchronize().unwrap();
            }
            let t0 = Instant::now();
            for _ in 0..200 {
                f(&ctx);
            }
            ctx.stream.synchronize().unwrap();
            t0.elapsed().as_secs_f64() * 1e3 / 200.0
        };
        let ms = time_one(&run);
        let ms_dw = time_one(&run_dw);
        eprintln!(
            "gemm bwd {label}: {ms:.3} ms/layer (dW+dX, x24 = {:.1} ms; dW alone {ms_dw:.3})",
            ms * 24.0
        );
    }
}
