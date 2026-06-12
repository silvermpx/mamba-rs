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

// ── Stage 5b: 64x64-tile TC twins ──────────────────────────────────────────

/// Forced-tile TC forward (Tile64 vs Tile128 on the same shape); returns Y
/// upcast to f32.
fn run_tc_tile(
    t: &Ctx,
    dt: WeightDtype,
    dims: (usize, usize, usize),
    data: (&[f32], &[f32], Option<&GpuBuffer>),
    tile: sgemm_bi::TcTile,
) -> Vec<f32> {
    let (m, _k, n) = dims;
    let (qx, qw, bias) = data;
    let xt = t.typed_buf(qx, dt);
    let wt = t.typed_buf(qw, dt);
    let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
    let ops = sgemm_bi::TcFwdOperands {
        y: TypedPtr {
            ptr: yt.cached_ptr(),
            dtype: dt,
        },
        x: TypedPtr {
            ptr: xt.cached_ptr(),
            dtype: dt,
        },
        w: TypedPtr {
            ptr: wt.cached_ptr(),
            dtype: dt,
        },
        bias_ptr: bias.map_or(0, |b| b.cached_ptr()),
    };
    sgemm_bi::sgemm_bi_forward_tc_with_tile(&t.ctx.stream, &t.ctx.kernels, &ops, dims, tile)
        .unwrap();
    t.ctx.stream.synchronize().unwrap();
    let mut out = vec![0.0f32; m * n];
    yt.download_f32(&t.ctx.stream, &mut out).unwrap();
    out
}

/// Launch-reality check (the 0.4.0 lesson): the TC64 kernels must have been
/// compiled with their own section-local geometry — 128 threads per CTA —
/// not inherited stale defines. A TC64 kernel that compiled with the
/// 256-thread geometry would reject its 128-thread launch (or worse,
/// silently mis-tile); asserting the function attribute catches the drift
/// at the source.
#[test]
fn tc64_kernel_geometry_is_128_threads() {
    let t = Ctx::new();
    let k = &t.ctx.kernels;
    let tc64 = [
        ("sgemm_bi_nn_tc64", &k.sgemm_nn_tc64_typed),
        ("sgemm_bi_tn_tc64", &k.sgemm_tn_tc64_typed),
        ("sgemm_bi_nt_tc64", &k.sgemm_nt_tc64_typed),
    ];
    for (name, kern) in tc64 {
        for (dt, f) in [("bf16", &kern.bf16), ("f16", &kern.f16)] {
            let mt = f.max_threads_per_block().unwrap();
            assert_eq!(mt, 128, "{name}_{dt}: MAX_THREADS_PER_BLOCK {mt} != 128");
        }
    }
    let tc128 = [
        ("sgemm_bi_nn_tc", &k.sgemm_nn_tc_typed),
        ("sgemm_bi_tn_tc", &k.sgemm_tn_tc_typed),
        ("sgemm_bi_nt_tc", &k.sgemm_nt_tc_typed),
    ];
    for (name, kern) in tc128 {
        for (dt, f) in [("bf16", &kern.bf16), ("f16", &kern.f16)] {
            let mt = f.max_threads_per_block().unwrap();
            assert_eq!(mt, 256, "{name}_{dt}: MAX_THREADS_PER_BLOCK {mt} != 256");
        }
    }
}

/// THE load-bearing test for the underfill-aware tile routing: the 64- and
/// 128-tile TC kernels must be BIT-IDENTICAL per output element (same
/// 32-wide reduction slabs, same ascending mma chain, same tail zero-fill).
/// Without this property an M-dependent tile pick would break the strict
/// all-M invariance contract.
#[test]
fn tc64_and_tc128_bit_identical() {
    use sgemm_bi::TcTile;
    let t = Ctx::new();
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, k, n) in [
            (256usize, 384usize, 512usize),
            (256, 100, 512),  // K-tail (K % 32 != 0)
            (300, 768, 1024), // M-tail
            (192, 384, 200),  // N-tail
        ] {
            let qx = quantize(&det(m * k, 11, 1.0), dt);
            let qw = quantize(&det(k * n, 22, 0.5), dt);
            let bias = det(n, 33, 0.25);
            let b32 = t.f32_buf(&bias);

            // fwd: forced Tile64 vs forced Tile128, with bias.
            let y64 = run_tc_tile(&t, dt, (m, k, n), (&qx, &qw, Some(&b32)), TcTile::Tile64);
            let y128 = run_tc_tile(&t, dt, (m, k, n), (&qx, &qw, Some(&b32)), TcTile::Tile128);
            for (i, (&a, &b)) in y64.iter().zip(&y128).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{dt:?} fwd M{m} K{k} N{n}: Tile64/Tile128 bit drift at {i}: {a:?} vs {b:?}"
                );
            }

            // dW: both accumulate f32 into a zeroed master.
            let qdy = quantize(&det(m * n, 55, 0.5), dt);
            let xt = t.typed_buf(&qx, dt);
            let dyt = t.typed_buf(&qdy, dt);
            let dytp = TypedPtr {
                ptr: dyt.cached_ptr(),
                dtype: dt,
            };
            let xtp = TypedPtr {
                ptr: xt.cached_ptr(),
                dtype: dt,
            };
            let mut dw_bits = Vec::new();
            for tile in [TcTile::Tile64, TcTile::Tile128] {
                let dw = GpuBuffer::zeros(&t.ctx.stream, k * n).unwrap();
                sgemm_bi::sgemm_bi_backward_dw_tc_with_tile(
                    &t.ctx.stream,
                    &t.ctx.kernels,
                    dw.cached_ptr(),
                    dytp,
                    xtp,
                    (m, k, n),
                    tile,
                )
                .unwrap();
                t.ctx.stream.synchronize().unwrap();
                dw_bits.push(dw.to_cpu(&t.ctx.stream).unwrap());
            }
            for (i, (a, b)) in dw_bits[0].iter().zip(&dw_bits[1]).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{dt:?} dW M{m} K{k} N{n}: Tile64/Tile128 bit drift at {i}"
                );
            }

            // dX: typed RNE overwrite.
            let wt = t.typed_buf(&qw, dt);
            let wtp = TypedPtr {
                ptr: wt.cached_ptr(),
                dtype: dt,
            };
            let mut dx_bits = Vec::new();
            for tile in [TcTile::Tile64, TcTile::Tile128] {
                let dxt = DtypedBuf::zeros(&t.ctx.stream, m * k, dt).unwrap();
                sgemm_bi::sgemm_bi_backward_dx_tc_with_tile(
                    &t.ctx.stream,
                    &t.ctx.kernels,
                    TypedPtr {
                        ptr: dxt.cached_ptr(),
                        dtype: dt,
                    },
                    dytp,
                    wtp,
                    (m, k, n),
                    tile,
                )
                .unwrap();
                t.ctx.stream.synchronize().unwrap();
                let mut got = vec![0.0f32; m * k];
                dxt.download_f32(&t.ctx.stream, &mut got).unwrap();
                dx_bits.push(got);
            }
            for (i, (a, b)) in dx_bits[0].iter().zip(&dx_bits[1]).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{dt:?} dX M{m} K{k} N{n}: Tile64/Tile128 bit drift at {i}"
                );
            }
        }
    }
}

/// TC64 small-shape accuracy vs the f32 reference (same class as the
/// 128-tile kernels): M/N in the [64, 128) band the 128 gate rejects,
/// plus K-tails and intra-tile M/N tails.
#[test]
fn tc64_forward_and_backward_match_f32_reference_small_shapes() {
    let t = Ctx::new();
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, k, n) in [
            (64usize, 384usize, 64usize),
            (96, 100, 96),  // K-tail + intra-tile M/N tails
            (127, 768, 65), // worst-case tile waste
            (64, 256, 512),
            (512, 256, 64),
        ] {
            let qx = quantize(&det(m * k, 11, 1.0), dt);
            let qw = quantize(&det(k * n, 22, 0.5), dt);
            let bias = det(n, 33, 0.25);
            let b32 = t.f32_buf(&bias);

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
            eprintln!("TC64 {dt:?} M{m} K{k} N{n}: cos vs f32 = {cos:.9}");
            assert!(
                cos > 0.9999,
                "TC64 {dt:?} M{m} K{k} N{n}: cos {cos} — fragment layout or staging bug"
            );

            // dW + dX through the auto-routed TC entries on the same shape.
            let qdy = quantize(&det(m * n, 55, 0.5), dt);
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
            let dw_want = dw_ref.to_cpu(&t.ctx.stream).unwrap();
            let dx_want = dx_ref.to_cpu(&t.ctx.stream).unwrap();

            let xt = t.typed_buf(&qx, dt);
            let dyt = t.typed_buf(&qdy, dt);
            let wt = t.typed_buf(&qw, dt);
            let dytp = TypedPtr {
                ptr: dyt.cached_ptr(),
                dtype: dt,
            };
            // dW gate keys on (K_out, N) = (k, n).
            if k >= 64 && n >= 64 {
                let dw_tc = GpuBuffer::zeros(&t.ctx.stream, k * n).unwrap();
                sgemm_bi::sgemm_bi_backward_dw_tc(
                    &t.ctx.stream,
                    &t.ctx.kernels,
                    dw_tc.cached_ptr(),
                    dytp,
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
                eprintln!("TC64 dW {dt:?} M{m} K{k} N{n}: cos vs f32 = {cos:.9}");
                assert!(cos > 0.9999, "TC64 dW {dt:?} M{m} K{k} N{n}: cos {cos}");
            }
            // dX gate keys on (M, K_out) = (m, k).
            let dxt = DtypedBuf::zeros(&t.ctx.stream, m * k, dt).unwrap();
            sgemm_bi::sgemm_bi_backward_dx_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                TypedPtr {
                    ptr: dxt.cached_ptr(),
                    dtype: dt,
                },
                dytp,
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
            eprintln!("TC64 dX {dt:?} M{m} K{k} N{n}: cos vs f32 = {cos:.9}");
            assert!(cos > 0.9999, "TC64 dX {dt:?} M{m} K{k} N{n}: cos {cos}");
        }
    }
}

/// Gate-boundary sweep with launch-reality routing asserts: every Ok must
/// report which tile actually launched, every below-gate shape must be the
/// exact `UNCOVERED`-prefixed Err the blas.rs fallback chain keys on.
#[test]
fn tc_route_gate_boundary_sweep() {
    use sgemm_bi::{TC64_PREFER_MAX_TILES128, TcTile};
    let t = Ctx::new();
    let dt = WeightDtype::Bf16;
    let k = 64usize;
    // N wide enough that M=128 lands at exactly the Tile128 threshold.
    let n_wide = 128 * TC64_PREFER_MAX_TILES128 as usize;

    let route = |m: usize, n: usize| -> Result<TcTile, String> {
        let qx = quantize(&det(m * k, 11, 1.0), dt);
        let qw = quantize(&det(k * n, 22, 0.5), dt);
        let xt = t.typed_buf(&qx, dt);
        let wt = t.typed_buf(&qw, dt);
        let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
        let r = sgemm_bi::sgemm_bi_forward_tc(
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
            0,
            (m, k, n),
        );
        t.ctx.stream.synchronize().unwrap();
        r
    };

    // Below the 64 gate -> honest UNCOVERED (blas.rs scalar fallback).
    for (m, n) in [(63usize, 64usize), (64, 63), (63, 4096), (4096, 63)] {
        let err = route(m, n).unwrap_err();
        assert!(
            err.starts_with("UNCOVERED"),
            "M{m} N{n}: expected UNCOVERED prefix, got: {err}"
        );
    }
    // [64, 128) band -> Tile64.
    assert_eq!(route(64, 64).unwrap(), TcTile::Tile64);
    assert_eq!(route(127, 127).unwrap(), TcTile::Tile64);
    assert_eq!(route(127, 4096).unwrap(), TcTile::Tile64);
    assert_eq!(route(4096, 127).unwrap(), TcTile::Tile64);
    // Both dims >= 128 but the 128-tile grid underfills -> Tile64.
    assert_eq!(route(128, 128).unwrap(), TcTile::Tile64);
    assert_eq!(route(1024, 512).unwrap(), TcTile::Tile64);
    // At/above the underfill threshold -> Tile128 (the big-shape kernels
    // keep their exact pre-existing routing and bits).
    assert_eq!(route(128, n_wide).unwrap(), TcTile::Tile128);
    assert_eq!(route(2048, 3072).unwrap(), TcTile::Tile128);

    // dW routes on (K_out, N), never on the reduction dim M.
    let dw_route = |kk: usize, n: usize| -> Result<TcTile, String> {
        let m = 256usize;
        let qx = quantize(&det(m * kk, 44, 1.0), dt);
        let qdy = quantize(&det(m * n, 55, 0.5), dt);
        let xt = t.typed_buf(&qx, dt);
        let dyt = t.typed_buf(&qdy, dt);
        let dw = GpuBuffer::zeros(&t.ctx.stream, kk * n).unwrap();
        let r = sgemm_bi::sgemm_bi_backward_dw_tc(
            &t.ctx.stream,
            &t.ctx.kernels,
            dw.cached_ptr(),
            TypedPtr {
                ptr: dyt.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: xt.cached_ptr(),
                dtype: dt,
            },
            (m, kk, n),
        );
        t.ctx.stream.synchronize().unwrap();
        r
    };
    assert!(dw_route(63, 512).unwrap_err().starts_with("UNCOVERED"));
    assert_eq!(dw_route(64, 64).unwrap(), TcTile::Tile64);
    assert_eq!(dw_route(128, 512).unwrap(), TcTile::Tile64); // d128 in_proj dW
    assert_eq!(dw_route(128, n_wide).unwrap(), TcTile::Tile128);

    // dX routes on (M, K_out).
    let dx_route = |m: usize, kk: usize| -> Result<TcTile, String> {
        let n = 256usize;
        let qdy = quantize(&det(m * n, 55, 0.5), dt);
        let qw = quantize(&det(kk * n, 77, 0.5), dt);
        let dyt = t.typed_buf(&qdy, dt);
        let wt = t.typed_buf(&qw, dt);
        let dxt = DtypedBuf::zeros(&t.ctx.stream, m * kk, dt).unwrap();
        let r = sgemm_bi::sgemm_bi_backward_dx_tc(
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
            (m, kk, n),
        );
        t.ctx.stream.synchronize().unwrap();
        r
    };
    assert!(dx_route(63, 512).unwrap_err().starts_with("UNCOVERED"));
    assert_eq!(dx_route(64, 64).unwrap(), TcTile::Tile64);
    assert_eq!(dx_route(1024, 128).unwrap(), TcTile::Tile64); // d128 in_proj dX
    assert_eq!(dx_route(9216, 128), Ok(TcTile::Tile128));
}

/// Strict all-M batch invariance of the TC fwd ACROSS the Tile64/Tile128
/// routing boundary: row 0 must be bit-identical for M in {64, 96, 127,
/// 128, 1024} (the first four route Tile64, M=1024 with N=3072 routes
/// Tile128). Also covers run-to-run determinism at a Tile64 shape.
#[test]
fn tc64_forward_strict_all_m_invariance_and_determinism() {
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

        // Run-to-run determinism at a Tile64-routed M.
        let a = run(96, 100);
        let b = run(96, 100);
        for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{dt:?} TC64 nondeterminism at col {i}"
            );
        }

        let y_ref = run(64, 200);
        for m in [96usize, 127, 128, 1024] {
            let y = run(m, 300 + m as u32);
            for (i, (&x, &y)) in y_ref.iter().zip(&y).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{dt:?} TC batch-variance at col {i}: M=64 vs M={m}"
                );
            }
        }
    }
}

#[test]
#[ignore] // wall-clock benchmark — run explicitly on a quiet GPU
fn bench_tc64_vs_tc128_small_shapes() {
    use sgemm_bi::TcTile;
    use std::time::Instant;
    let t = Ctx::new();
    let dt = WeightDtype::Bf16;
    // (m, k, n, label) — trainer GEMMs of the d128/d256 benches plus the
    // crossover region (tiles128 between 32 and 192).
    for (m, k, n, label) in [
        (1024usize, 128usize, 512usize, "d128 in_proj fwd (t128=32)"),
        (1024, 256, 128, "d128 out_proj fwd (t128=8)"),
        (2048, 256, 1024, "d256 in_proj fwd (t128=128)"),
        (2048, 512, 256, "d256 out_proj fwd (t128=32)"),
        (2048, 768, 512, "crossover (t128=64)"),
        (2048, 1536, 768, "d768 out_proj fwd (t128=96)"),
        (2048, 768, 3072, "d768 in_proj fwd (t128=384)"),
    ] {
        let qx = quantize(&det(m * k, 11, 1.0), dt);
        let qw = quantize(&det(k * n, 22, 0.5), dt);
        let qdy = quantize(&det(m * n, 55, 0.5), dt);
        let xt = t.typed_buf(&qx, dt);
        let wt = t.typed_buf(&qw, dt);
        let dyt = t.typed_buf(&qdy, dt);
        let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
        let dw = GpuBuffer::zeros(&t.ctx.stream, k * n).unwrap();
        let dxt = DtypedBuf::zeros(&t.ctx.stream, m * k, dt).unwrap();
        let tp = |b: &DtypedBuf| TypedPtr {
            ptr: b.cached_ptr(),
            dtype: dt,
        };
        let (xtp, wtp, dytp, ytp, dxtp) = (tp(&xt), tp(&wt), tp(&dyt), tp(&yt), tp(&dxt));

        let iters = 200;
        let time_path = |label: &str, f: &dyn Fn()| -> f64 {
            for _ in 0..5 {
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

        eprintln!("[{label}] M{m} K{k} N{n}");
        let ops = sgemm_bi::TcFwdOperands {
            y: ytp,
            x: xtp,
            w: wtp,
            bias_ptr: 0,
        };
        for tile in [TcTile::Tile64, TcTile::Tile128] {
            time_path(&format!("fwd {tile:?}"), &|| {
                sgemm_bi::sgemm_bi_forward_tc_with_tile(
                    &t.ctx.stream,
                    &t.ctx.kernels,
                    &ops,
                    (m, k, n),
                    tile,
                )
                .unwrap();
            });
        }
        for tile in [TcTile::Tile64, TcTile::Tile128] {
            time_path(&format!("dW {tile:?}"), &|| {
                sgemm_bi::sgemm_bi_backward_dw_tc_with_tile(
                    &t.ctx.stream,
                    &t.ctx.kernels,
                    dw.cached_ptr(),
                    dytp,
                    xtp,
                    (m, k, n),
                    tile,
                )
                .unwrap();
            });
        }
        for tile in [TcTile::Tile64, TcTile::Tile128] {
            time_path(&format!("dX {tile:?}"), &|| {
                sgemm_bi::sgemm_bi_backward_dx_tc_with_tile(
                    &t.ctx.stream,
                    &t.ctx.kernels,
                    dxtp,
                    dytp,
                    wtp,
                    (m, k, n),
                    tile,
                )
                .unwrap();
            });
        }
        // Scalar-tier reference on the same shape (what TC64 must beat).
        use mamba_rs::mamba_ssm::gpu::blas::{
            bi_sgemm_backward_dw_typed, bi_sgemm_backward_dx_typed, bi_sgemm_forward_typed,
        };
        time_path("fwd scalar bi", &|| {
            bi_sgemm_forward_typed(&t.ctx, ytp, xtp, wtp, 0, (m, k, n)).unwrap();
        });
        time_path("dW scalar bi", &|| {
            bi_sgemm_backward_dw_typed(&t.ctx, dw.cached_ptr(), dytp, xtp, (m, k, n)).unwrap();
        });
        time_path("dX scalar bi", &|| {
            bi_sgemm_backward_dx_typed(&t.ctx, dxtp, dytp, wtp, (m, k, n)).unwrap();
        });
    }
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

/// Step-0 instrumentation for the BK=64 squeeze (internal/tc-bk64-blueprint.md):
/// function attributes (regs / spills / static smem), bit-level golden hashes
/// of the full TC triad on the contract shapes, and baseline fwd timings.
/// Run before and after each blueprint step; goldens must not move on steps
/// that promise bit-identity (swizzle, dynsmem, BK=64 on K%64 ∉ (0,32] shapes).
#[test]
#[ignore]
fn step0_tc_attrs_and_goldens() {
    use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
    use std::time::Instant;
    let t = Ctx::new();
    let k = &t.ctx.kernels;

    println!("== function attributes ==");
    let fams = [
        ("nn_tc", &k.sgemm_nn_tc_typed),
        ("tn_tc", &k.sgemm_tn_tc_typed),
        ("nt_tc", &k.sgemm_nt_tc_typed),
        ("nn_tc64", &k.sgemm_nn_tc64_typed),
        ("tn_tc64", &k.sgemm_tn_tc64_typed),
        ("nt_tc64", &k.sgemm_nt_tc64_typed),
    ];
    for (name, kern) in fams {
        for (dtn, f) in [("bf16", &kern.bf16), ("f16", &kern.f16)] {
            println!(
                "{name}_{dtn}: regs={} local={} smem={} maxthr={}",
                f.num_regs().unwrap(),
                f.local_size_bytes().unwrap(),
                f.shared_size_bytes().unwrap(),
                f.max_threads_per_block().unwrap(),
            );
        }
    }

    fn fnv(bits: impl Iterator<Item = u32>) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for b in bits {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    println!("== golden hashes (fwd/dW/dX bits) ==");
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, k_, n) in [
            (256usize, 384usize, 512usize),
            (256, 100, 512),
            (300, 768, 3072),
            (2048, 768, 3072),
        ] {
            let qx = quantize(&det(m * k_, 11, 1.0), dt);
            let qw = quantize(&det(k_ * n, 22, 0.5), dt);
            let bias = det(n, 33, 0.25);
            let b32 = t.f32_buf(&bias);
            let y = run_tc(&t, dt, (m, k_, n), &qx, &qw, Some(&b32));
            let h_fwd = fnv(y.iter().map(|v| v.to_bits()));

            let qdy = quantize(&det(m * n, 44, 0.5), dt);
            let dyt = t.typed_buf(&qdy, dt);
            let xt = t.typed_buf(&qx, dt);
            let wt = t.typed_buf(&qw, dt);

            let dw = GpuBuffer::zeros(&t.ctx.stream, k_ * n).unwrap();
            sgemm_bi::sgemm_bi_backward_dw_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                dw.cached_ptr(),
                TypedPtr {
                    ptr: dyt.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: xt.cached_ptr(),
                    dtype: dt,
                },
                (m, k_, n),
            )
            .unwrap();
            let dxt = DtypedBuf::zeros(&t.ctx.stream, m * k_, dt).unwrap();
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
                (m, k_, n),
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let dwh = dw.to_cpu(&t.ctx.stream).unwrap();
            let mut dxh = vec![0.0f32; m * k_];
            dxt.download_f32(&t.ctx.stream, &mut dxh).unwrap();
            println!(
                "[{:?} M{m} K{k_} N{n}] fwd={:016x} dW={:016x} dX={:016x}",
                dt,
                h_fwd,
                fnv(dwh.iter().map(|v| v.to_bits())),
                fnv(dxh.iter().map(|v| v.to_bits())),
            );
        }
    }

    println!("== baseline fwd timings (bf16, 128-tile route) ==");
    for (m, k_, n) in [(2048usize, 768usize, 3072usize), (4096, 1536, 3072)] {
        let qx = quantize(&det(m * k_, 11, 1.0), WeightDtype::Bf16);
        let qw = quantize(&det(k_ * n, 22, 0.5), WeightDtype::Bf16);
        let xt = t.typed_buf(&qx, WeightDtype::Bf16);
        let wt = t.typed_buf(&qw, WeightDtype::Bf16);
        let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, WeightDtype::Bf16).unwrap();
        let run = || {
            sgemm_bi::sgemm_bi_forward_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                TypedPtr {
                    ptr: yt.cached_ptr(),
                    dtype: WeightDtype::Bf16,
                },
                TypedPtr {
                    ptr: xt.cached_ptr(),
                    dtype: WeightDtype::Bf16,
                },
                TypedPtr {
                    ptr: wt.cached_ptr(),
                    dtype: WeightDtype::Bf16,
                },
                0,
                (m, k_, n),
            )
            .unwrap();
        };
        for _ in 0..3 {
            run();
        }
        t.ctx.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..50 {
            run();
        }
        t.ctx.stream.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / 50.0;
        let tflops = 2.0 * m as f64 * k_ as f64 * n as f64 / (us * 1e-6) / 1e12;
        println!("fwd M{m} K{k_} N{n}: {us:.1} us = {tflops:.1} TFLOPS");
    }
}
