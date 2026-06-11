//! Batch-invariant SGEMM triad (sgemm_bi) — determinism, parity, and speed.
//!
//! With `ctx.set_batch_invariant(true)` every f32 training GEMM (NN fwd,
//! TN dW, NT dX) routes through the deterministic warptiling dispatcher in
//! `gpu/sgemm_bi.rs` instead of cuBLAS TF32. Contract under test:
//!
//! 1. `flag_on_training_is_bit_identical_across_runs` — two fresh f32
//!    trainers, same seed/inputs, N steps each: snapshots must agree BIT
//!    FOR BIT (cuBLAS gives only ~1e-5 agreement across instances because
//!    its algorithm heuristics may differ).
//! 2. `flag_on_matches_cublas_loosely` — the deterministic path computes
//!    the same math as cuBLAS f32/TF32 up to TF32 noise: weight deltas
//!    after a step agree within loose tolerances (catches arg-order /
//!    transpose / accumulation bugs in the ported dispatcher).
//! 3. `nn_forward_is_batch_invariant_within_bucket` — `Y[0,:]` is
//!    bit-identical across batch sizes that route to the same dispatch
//!    bucket (M=1 vs M=16 → ultra-thin; M=64 vs M=256 → split-K32).
//!    Crossing a bucket boundary (e.g. M=31 → M=32) may change the
//!    K-reduction association by design — every bucket stays
//!    deterministic, but the buckets are distinct fixed orders. Strict
//!    all-M invariance is the INFERENCE matvec_bi kernel's contract.
//! 4. `bench_sgemm_bi_vs_tf32` (#[ignore]) — wall-clock of trainer steps
//!    with the flag on vs off across small/medium/large shapes.

#![cfg(feature = "cuda")]

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
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

fn cfg() -> MambaConfig {
    MambaConfig {
        d_model: 128,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    }
}

fn flat_weights(w: &MambaWeights) -> Vec<f32> {
    let mut out = Vec::new();
    out.extend_from_slice(&w.input_proj_w);
    out.extend_from_slice(&w.input_proj_b);
    for l in &w.layers {
        out.extend_from_slice(&l.norm_weight);
        out.extend_from_slice(&l.in_proj_w);
        out.extend_from_slice(&l.conv1d_weight);
        out.extend_from_slice(&l.conv1d_bias);
        out.extend_from_slice(&l.x_proj_w);
        out.extend_from_slice(&l.dt_proj_w);
        out.extend_from_slice(&l.dt_proj_b);
        out.extend_from_slice(&l.a_log);
        out.extend_from_slice(&l.d_param);
        out.extend_from_slice(&l.out_proj_w);
    }
    out.extend_from_slice(&w.norm_f_weight);
    out
}

/// Build an f32 trainer with the batch-invariant flag set as requested and
/// run `steps` identical training steps. Returns the final master snapshot.
fn run_training(batch: usize, seq_len: usize, steps: usize, invariant: bool) -> Vec<f32> {
    let cfg = cfg();
    let input_dim = cfg.d_model;
    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xDE7E_4213);
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    let session = TrainSessionCfg {
        input_dim,
        batch,
        seq_len,
        lr: 1e-3,
        weight_decay: 0.0,
    };
    let mut trainer = MambaTrainer::new_full(0, &cpu, cfg, session, WeightDtype::F32)
        .expect("construct f32 trainer");
    trainer.ctx().set_batch_invariant(invariant);

    let n = batch * seq_len * input_dim;
    for s in 0..steps {
        trainer
            .step(&det(n, 0x11 + s as u32, 1.0), &det(n, 0x77 + s as u32, 0.1))
            .expect("step");
    }
    flat_weights(&trainer.snapshot_master().expect("snapshot"))
}

#[test]
fn flag_on_training_is_bit_identical_across_runs() {
    let a = run_training(4, 64, 5, true);
    let b = run_training(4, 64, 5, true);
    assert_eq!(a.len(), b.len());
    let mut diffs = 0usize;
    for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
        if x.to_bits() != y.to_bits() {
            if diffs < 5 {
                eprintln!("bit mismatch at {i}: {x:?} vs {y:?}");
            }
            diffs += 1;
        }
    }
    assert_eq!(
        diffs, 0,
        "batch-invariant training must be bit-identical across runs"
    );
}

#[test]
fn flag_on_matches_cublas_loosely() {
    // Same training trajectory with the deterministic triad vs cuBLAS TF32.
    // TF32's 10-bit mantissa compounds through 5 steps; this is a sanity
    // bound against transpose/accumulation bugs, not a precision claim.
    let inv = run_training(4, 64, 5, true);
    let blas = run_training(4, 64, 5, false);
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in inv.iter().zip(&blas) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
    eprintln!("sgemm_bi vs cuBLAS-TF32 snapshot cosine = {cos:.9}");
    assert!(
        cos > 0.99999,
        "deterministic triad diverged from cuBLAS trajectory: cos={cos}"
    );
}

#[test]
fn nn_forward_is_batch_invariant_within_bucket() {
    use mamba_rs::mamba_ssm::gpu::blas::gpu_sgemm_forward_raw;
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;

    let device = GpuDevice::new(0).expect("gpu");
    let ctx = GpuCtx::new(&device).expect("ctx");
    ctx.set_batch_invariant(true);

    let (k, n) = (384, 512);
    let row = det(k, 42, 1.0);
    let w_host = det(k * n, 43, 0.5);

    let run = |m: usize| -> Vec<f32> {
        // Row 0 = `row`; other rows vary (they must not affect row 0).
        let mut x_host = det(m * k, 100 + m as u32, 1.0);
        x_host[..k].copy_from_slice(&row);
        let mut x = GpuBuffer::zeros(&ctx.stream, m * k).unwrap();
        x.upload(&ctx.stream, &x_host).unwrap();
        let mut w = GpuBuffer::zeros(&ctx.stream, k * n).unwrap();
        w.upload(&ctx.stream, &w_host).unwrap();
        let mut y = GpuBuffer::zeros(&ctx.stream, m * n).unwrap();
        gpu_sgemm_forward_raw(&ctx, &mut y, &x, w.cached_ptr(), None, (m, k, n)).unwrap();
        ctx.stream.synchronize().unwrap();
        let host = y.to_cpu(&ctx.stream).unwrap();
        host[..n].to_vec()
    };

    // Ultra-thin bucket: M in [1, 32).
    let y1 = run(1);
    let y16 = run(16);
    for (i, (&a, &b)) in y1.iter().zip(&y16).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "batch-variance in ultra-thin bucket at col {i}: M=1 {a:?} vs M=16 {b:?}"
        );
    }
    // Split-K32 bucket: M in [32, 1024], K % 32 == 0, N in [128, 2048].
    let y64 = run(64);
    let y256 = run(256);
    for (i, (&a, &b)) in y64.iter().zip(&y256).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "batch-variance in split-K bucket at col {i}: M=64 {a:?} vs M=256 {b:?}"
        );
    }
}

#[test]
#[ignore] // wall-clock benchmark — run explicitly on a quiet GPU
fn bench_sgemm_bi_vs_tf32() {
    use std::time::Instant;
    // (d_model, n_layers, batch, seq_len, label)
    let shapes = [
        (128usize, 2usize, 16usize, 64usize, "RL-small d128"),
        (256, 4, 16, 128, "d256"),
        (768, 4, 8, 256, "130m-ish d768"),
        (1536, 2, 4, 256, "770m-ish d1536"),
    ];
    for (dm, nl, b, t, label) in shapes {
        let cfg = MambaConfig {
            d_model: dm,
            n_layers: nl,
            ..cfg()
        };
        let input_dim = cfg.d_model;
        let mut cpu = MambaWeights::init(&cfg, input_dim, 7);
        for lw in cpu.layers.iter_mut() {
            lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
        }
        let session = TrainSessionCfg {
            input_dim,
            batch: b,
            seq_len: t,
            lr: 1e-4,
            weight_decay: 0.0,
        };
        let n = b * t * input_dim;
        let input = det(n, 1, 1.0);
        let dtemp = det(n, 2, 0.1);

        let time_mode = |invariant: bool| -> f64 {
            let mut tr =
                MambaTrainer::new_full(0, &cpu, cfg, session, WeightDtype::F32).expect("trainer");
            tr.ctx().set_batch_invariant(invariant);
            for _ in 0..3 {
                tr.step(&input, &dtemp).expect("warmup");
            }
            let iters = 20;
            let start = Instant::now();
            for _ in 0..iters {
                tr.step(&input, &dtemp).expect("step");
            }
            start.elapsed().as_secs_f64() / iters as f64
        };

        let t_blas = time_mode(false);
        let t_bi = time_mode(true);
        eprintln!(
            "[{label}] B={b} T={t}: cuBLAS-TF32 {:.3} ms/step | sgemm_bi {:.3} ms/step | ratio {:.2}x",
            t_blas * 1e3,
            t_bi * 1e3,
            t_bi / t_blas
        );
    }
}
