//! Stage 5 tensor-core tier (`bi_tensor_cores`) — contract tests.
//!
//! The TC NN forward (`sgemm_bi_nn_tc_*`, mma.sync.m16n8k16 + f32
//! accumulate) is a SEPARATE numeric contract from the scalar triad: its
//! reduction tree differs from the ascending-K FMA chain, so outputs do not
//! bit-match the scalar kernels. What it MUST satisfy:
//!   1. correctness — close to the f32 reference on quantized inputs
//!      (cosine; a fragment-layout bug shows up as garbage, not noise);
//!   2. run-to-run determinism — bit-identical across launches;
//!   3. STRICT batch invariance — row 0 bit-identical across ALL M (each
//!      element's K-reduction lives in one warp, independent of grid).

#![cfg(feature = "cuda")]

use half::{bf16, f16};
use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::sgemm_bi;

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

fn quantize(v: &[f32], dt: WeightDtype) -> Vec<f32> {
    match dt {
        WeightDtype::Bf16 => v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect(),
        WeightDtype::F16 => v.iter().map(|&x| f16::from_f32(x).to_f32()).collect(),
        WeightDtype::F32 => v.to_vec(),
    }
}

struct Ctx {
    ctx: GpuCtx,
}

impl Ctx {
    fn new() -> Self {
        let device = GpuDevice::new(0).expect("gpu");
        let ctx = GpuCtx::new(&device).expect("ctx");
        Self { ctx }
    }

    fn typed_buf(&self, data: &[f32], dt: WeightDtype) -> DtypedBuf {
        let b = DtypedBuf::zeros(&self.ctx.stream, data.len(), dt).unwrap();
        b.upload_f32(&self.ctx.stream, data).unwrap();
        b
    }

    fn f32_buf(&self, data: &[f32]) -> GpuBuffer {
        let mut b = GpuBuffer::zeros(&self.ctx.stream, data.len()).unwrap();
        b.upload(&self.ctx.stream, data).unwrap();
        b
    }
}

/// Launch the TC forward on quantized inputs; returns Y upcast to f32.
fn run_tc(
    t: &Ctx,
    dt: WeightDtype,
    dims: (usize, usize, usize),
    qx: &[f32],
    qw: &[f32],
    bias: Option<&GpuBuffer>,
) -> Vec<f32> {
    let (m, _k, n) = dims;
    let xt = t.typed_buf(qx, dt);
    let wt = t.typed_buf(qw, dt);
    let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
    sgemm_bi::sgemm_bi_forward_tc(
        &t.ctx.stream,
        &t.ctx.kernels,
        TypedPtr {
            ptr: yt.cached_ptr(),
            dtype: dt,
        },
        TypedPtr {
            ptr: xt.cached_ptr(),
            dtype: dt,
        },
        TypedPtr {
            ptr: wt.cached_ptr(),
            dtype: dt,
        },
        bias.map_or(0, |b| b.cached_ptr()),
        dims,
    )
    .unwrap();
    t.ctx.stream.synchronize().unwrap();
    let mut out = vec![0.0f32; m * n];
    yt.download_f32(&t.ctx.stream, &mut out).unwrap();
    out
}

fn cos_sim(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

#[test]
fn tc_forward_matches_f32_reference_loosely() {
    let t = Ctx::new();
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, k, n) in [
            (256usize, 384usize, 512usize),
            (256, 100, 512),  // K-tail (K % 32 != 0)
            (300, 768, 3072), // M/N-tails
            (2048, 768, 3072),
        ] {
            let qx = quantize(&det(m * k, 11, 1.0), dt);
            let qw = quantize(&det(k * n, 22, 0.5), dt);
            let bias = det(n, 33, 0.25);
            let b32 = t.f32_buf(&bias);

            // f32 reference on the SAME quantized values.
            let x32 = t.f32_buf(&qx);
            let w32 = t.f32_buf(&qw);
            let mut y32 = GpuBuffer::zeros(&t.ctx.stream, m * n).unwrap();
            sgemm_bi::sgemm_bi_forward(
                &t.ctx.stream,
                &t.ctx.kernels,
                &mut y32,
                &x32,
                w32.cached_ptr(),
                b32.cached_ptr(),
                (m, k, n),
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let reference = y32.to_cpu(&t.ctx.stream).unwrap();

            let got = run_tc(&t, dt, (m, k, n), &qx, &qw, Some(&b32));
            let cos = cos_sim(&got, &reference);
            eprintln!("TC {dt:?} M{m} K{k} N{n}: cos vs f32 = {cos:.9}");
            assert!(
                cos > 0.9999,
                "TC {dt:?} M{m} K{k} N{n}: cos {cos} — fragment layout or staging bug"
            );
        }
    }
}

#[test]
fn tc_forward_is_deterministic_and_all_m_batch_invariant() {
    let t = Ctx::new();
    let (k, n) = (768usize, 3072usize);
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        let row = quantize(&det(k, 42, 1.0), dt);
        let qw = quantize(&det(k * n, 43, 0.5), dt);

        let run = |m: usize, seed: u32| -> Vec<f32> {
            let mut qx = quantize(&det(m * k, seed, 1.0), dt);
            qx[..k].copy_from_slice(&row);
            run_tc(&t, dt, (m, k, n), &qx, &qw, None)[..n].to_vec()
        };

        // Run-to-run determinism at fixed M.
        let a = run(256, 100);
        let b = run(256, 100);
        for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{dt:?} TC nondeterminism at col {i}"
            );
        }

        // STRICT all-M batch invariance of row 0 (other rows differ).
        let y128 = run(128, 200);
        let y512 = run(512, 300);
        let y2048 = run(2048, 400);
        for (i, (&x, &y)) in y128.iter().zip(&y512).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{dt:?} TC batch-variance at col {i}: M=128 vs M=512"
            );
        }
        for (i, (&x, &y)) in y128.iter().zip(&y2048).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{dt:?} TC batch-variance at col {i}: M=128 vs M=2048"
            );
        }
    }
}

#[test]
fn tc_backward_matches_f32_reference_loosely() {
    let t = Ctx::new();
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, k, n) in [
            (256usize, 384usize, 512usize),
            (300, 768, 3072), // tails on every axis
            (2048, 768, 512),
        ] {
            let qx = quantize(&det(m * k, 44, 1.0), dt);
            let qdy = quantize(&det(m * n, 55, 0.5), dt);
            let qw = quantize(&det(k * n, 77, 0.5), dt);

            // --- dW: f32 reference vs TC, both accumulate into f32 ---
            let x32 = t.f32_buf(&qx);
            let dy32 = t.f32_buf(&qdy);
            let dw_ref = GpuBuffer::zeros(&t.ctx.stream, k * n).unwrap();
            sgemm_bi::sgemm_bi_backward_dw(
                &t.ctx.stream,
                &t.ctx.kernels,
                dw_ref.cached_ptr(),
                &dy32,
                &x32,
                (m, k, n),
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let dw_want = dw_ref.to_cpu(&t.ctx.stream).unwrap();

            let xt = t.typed_buf(&qx, dt);
            let dyt = t.typed_buf(&qdy, dt);
            let dw_tc = GpuBuffer::zeros(&t.ctx.stream, k * n).unwrap();
            sgemm_bi::sgemm_bi_backward_dw_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                dw_tc.cached_ptr(),
                TypedPtr {
                    ptr: dyt.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: xt.cached_ptr(),
                    dtype: dt,
                },
                (m, k, n),
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let dw_got = dw_tc.to_cpu(&t.ctx.stream).unwrap();
            let cos = cos_sim(&dw_got, &dw_want);
            eprintln!("TC dW {dt:?} M{m} K{k} N{n}: cos vs f32 = {cos:.9}");
            assert!(cos > 0.9999, "TC dW {dt:?} M{m} K{k} N{n}: cos {cos}");

            // --- dX: typed output vs f32 reference ---
            let w32 = t.f32_buf(&qw);
            let mut dx_ref = GpuBuffer::zeros(&t.ctx.stream, m * k).unwrap();
            sgemm_bi::sgemm_bi_backward_dx(
                &t.ctx.stream,
                &t.ctx.kernels,
                &mut dx_ref,
                &dy32,
                w32.cached_ptr(),
                (m, k, n),
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let dx_want = dx_ref.to_cpu(&t.ctx.stream).unwrap();

            let wt = t.typed_buf(&qw, dt);
            let dxt = DtypedBuf::zeros(&t.ctx.stream, m * k, dt).unwrap();
            sgemm_bi::sgemm_bi_backward_dx_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                TypedPtr {
                    ptr: dxt.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: dyt.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: wt.cached_ptr(),
                    dtype: dt,
                },
                (m, k, n),
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let mut dx_got = vec![0.0f32; m * k];
            dxt.download_f32(&t.ctx.stream, &mut dx_got).unwrap();
            let cos = cos_sim(&dx_got, &dx_want);
            eprintln!("TC dX {dt:?} M{m} K{k} N{n}: cos vs f32 = {cos:.9}");
            assert!(cos > 0.9999, "TC dX {dt:?} M{m} K{k} N{n}: cos {cos}");
        }
    }
}

#[test]
fn tc_mixed_training_is_bit_identical_across_runs() {
    // End-to-end: bf16 mixed trainer with BOTH flags on. The TC contract
    // is different bits than the scalar tier, but it must still be
    // bit-identical across fresh runs (incl. CUDA Graph capture/replay).
    use mamba_rs::config::{MambaConfig, ScanMode};
    use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
    use mamba_rs::weights::MambaWeights;

    let cfg = MambaConfig {
        d_model: 128,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    };
    let run = || -> Vec<f32> {
        let mut cpu = MambaWeights::init(&cfg, cfg.d_model, 0xDE7E_4213);
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
        for lw in cpu.layers.iter_mut() {
            lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
        }
        let session = TrainSessionCfg {
            input_dim: cfg.d_model,
            batch: 4,
            seq_len: 64,
            lr: 1e-3,
            weight_decay: 0.0,
        };
        let mut tr =
            MambaTrainer::new_full(0, &cpu, cfg, session, WeightDtype::Bf16).expect("trainer");
        tr.ctx().set_batch_invariant(true);
        tr.ctx().set_bi_tensor_cores(true);
        let n = 4 * 64 * cfg.d_model;
        for s in 0..5 {
            tr.step(&det(n, 0x11 + s as u32, 1.0), &det(n, 0x77 + s as u32, 0.1))
                .expect("step");
        }
        let w = tr.snapshot_master().expect("snapshot");
        let mut out = Vec::new();
        for l in &w.layers {
            out.extend_from_slice(&l.in_proj_w);
            out.extend_from_slice(&l.out_proj_w);
        }
        out
    };
    let a = run();
    let b = run();
    let diffs = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(
        diffs, 0,
        "TC mixed training must be bit-identical across runs"
    );
}

#[test]
#[ignore] // wall-clock benchmark — run explicitly on a quiet GPU
fn bench_tc_vs_scalar_paths() {
    use mamba_rs::mamba_ssm::gpu::blas::bi_sgemm_forward_typed;
    use std::time::Instant;
    let t = Ctx::new();
    let dt = WeightDtype::Bf16;
    for (m, k, n) in [
        (2048usize, 768usize, 3072usize),
        (4096, 1536, 3072),
        (2048, 768, 512),
    ] {
        let qx = quantize(&det(m * k, 11, 1.0), dt);
        let qw = quantize(&det(k * n, 22, 0.5), dt);
        let xt = t.typed_buf(&qx, dt);
        let wt = t.typed_buf(&qw, dt);
        let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
        let (ytp, xtp, wtp) = (
            TypedPtr {
                ptr: yt.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: xt.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: wt.cached_ptr(),
                dtype: dt,
            },
        );

        let iters = 50;
        let time_path = |label: &str, f: &dyn Fn()| -> f64 {
            for _ in 0..3 {
                f();
            }
            t.ctx.stream.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.ctx.stream.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            eprintln!("  {label}: {us:.1} us");
            us
        };

        eprintln!("[M{m} K{k} N{n}]");
        let scalar = time_path("scalar bi (native/fallback)", &|| {
            bi_sgemm_forward_typed(&t.ctx, ytp, xtp, wtp, 0, (m, k, n)).unwrap();
        });
        let tc = time_path("tensor-core bi", &|| {
            sgemm_bi::sgemm_bi_forward_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                ytp,
                xtp,
                wtp,
                0,
                (m, k, n),
            )
            .unwrap();
        });
        eprintln!("  TC speedup vs scalar bi: {:.2}x", scalar / tc);

        // Backward twins on the same shape.
        use mamba_rs::mamba_ssm::gpu::blas::{
            bi_sgemm_backward_dw_typed, bi_sgemm_backward_dx_typed,
        };
        let qdy = quantize(&det(m * n, 55, 0.5), dt);
        let dyt = t.typed_buf(&qdy, dt);
        let dytp = TypedPtr {
            ptr: dyt.cached_ptr(),
            dtype: dt,
        };
        let dw = GpuBuffer::zeros(&t.ctx.stream, k * n).unwrap();
        let dxt = DtypedBuf::zeros(&t.ctx.stream, m * k, dt).unwrap();
        let dxtp = TypedPtr {
            ptr: dxt.cached_ptr(),
            dtype: dt,
        };
        let dw_s = time_path("dW scalar bi", &|| {
            bi_sgemm_backward_dw_typed(&t.ctx, dw.cached_ptr(), dytp, xtp, (m, k, n)).unwrap();
        });
        let dw_tc = time_path("dW tensor-core", &|| {
            sgemm_bi::sgemm_bi_backward_dw_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                dw.cached_ptr(),
                dytp,
                xtp,
                (m, k, n),
            )
            .unwrap();
        });
        eprintln!("  dW TC speedup: {:.2}x", dw_s / dw_tc);
        let dx_s = time_path("dX scalar bi", &|| {
            bi_sgemm_backward_dx_typed(&t.ctx, dxtp, dytp, wtp, (m, k, n)).unwrap();
        });
        let dx_tc = time_path("dX tensor-core", &|| {
            sgemm_bi::sgemm_bi_backward_dx_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                dxtp,
                dytp,
                wtp,
                (m, k, n),
            )
            .unwrap();
        });
        eprintln!("  dX TC speedup: {:.2}x", dx_s / dx_tc);
    }
}
