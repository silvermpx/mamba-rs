//! Stage 5 tensor-core tier (`bi_tensor_cores`) — contract tests.
//!
//! The TC NN forward (`gemm_bi_nn_tc_*`, mma.sync.m16n8k16 + f32
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
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad;

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
    gemm_bi_triad::gemm_bi_forward_tc(
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
            gemm_bi_triad::gemm_bi_forward(
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
#[ignore = "requires an SM89 GPU and exercises the deep Split-K policy boundary"]
fn sm89_deep_split_k_policy_is_bitwise_stable_and_keeps_forced_tc() {
    use mamba_rs::mamba_ssm::gpu::blas::gemm_bi_forward_typed;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::TcTile;

    let device = GpuDevice::new(0).expect("gpu");
    assert_eq!(device.compute_capability, (8, 9), "exact SM89 gate");
    let t = Ctx::new();
    t.ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();

    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for dims in [(128usize, 511usize, 128usize), (128, 1024, 128)] {
            let (m, k, n) = dims;
            let qx = quantize(&det(m * k, 0x8911, 1.0), dt);
            let qw = quantize(&det(k * n, 0x8922, 0.5), dt);
            let xt = t.typed_buf(&qx, dt);
            let wt = t.typed_buf(&qw, dt);
            let policy_output = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
            let direct_output = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
            let forced_output = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
            let typed = |buffer: &DtypedBuf| TypedPtr {
                ptr: buffer.cached_ptr(),
                dtype: dt,
            };
            let run_policy = |tensor_cores: bool| {
                t.ctx.set_bi_tensor_cores(tensor_cores);
                gemm_bi_forward_typed(
                    &t.ctx,
                    typed(&policy_output),
                    typed(&xt),
                    typed(&wt),
                    0,
                    dims,
                )
                .unwrap();
                t.ctx.stream.synchronize().unwrap();
                let mut values = vec![0.0f32; m * n];
                policy_output
                    .download_f32(&t.ctx.stream, &mut values)
                    .unwrap();
                values.into_iter().map(f32::to_bits).collect::<Vec<_>>()
            };

            let scalar = run_policy(false);
            let automatic = run_policy(true);
            assert_eq!(automatic, scalar, "{dt:?} automatic scalar route {dims:?}");
            for repeat in 0..20 {
                assert_eq!(
                    run_policy(true),
                    automatic,
                    "{dt:?} automatic repeat {repeat} at {dims:?}"
                );
            }

            let tile = gemm_bi_triad::gemm_bi_forward_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                typed(&direct_output),
                typed(&xt),
                typed(&wt),
                0,
                dims,
            )
            .unwrap();
            assert_eq!(tile, TcTile::Tile64, "direct TC route {dims:?}");
            let forced_ops = gemm_bi_triad::TcFwdOperands {
                y: typed(&forced_output),
                x: typed(&xt),
                w: typed(&wt),
                bias_ptr: 0,
            };
            gemm_bi_triad::gemm_bi_forward_tc_with_tile(
                &t.ctx.stream,
                &t.ctx.kernels,
                &forced_ops,
                dims,
                TcTile::Tile64,
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let mut direct = vec![0.0f32; m * n];
            let mut forced = vec![0.0f32; m * n];
            direct_output
                .download_f32(&t.ctx.stream, &mut direct)
                .unwrap();
            forced_output
                .download_f32(&t.ctx.stream, &mut forced)
                .unwrap();
            assert_eq!(
                direct
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                forced
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{dt:?} direct and forced Tile64 bits at {dims:?}"
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
            gemm_bi_triad::gemm_bi_backward_dw(
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
            gemm_bi_triad::gemm_bi_backward_dw_tc(
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
            gemm_bi_triad::gemm_bi_backward_dx(
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
            gemm_bi_triad::gemm_bi_backward_dx_tc(
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
    use mamba_rs::config::{MambaConfig, ScanMode};
    use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
    use mamba_rs::weights::MambaWeights;

    fn digest(values: &[f32]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in values {
            for byte in value.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    fn master_digest(weights: &MambaWeights) -> u64 {
        let mut values = Vec::new();
        values.extend_from_slice(&weights.input_proj_w);
        values.extend_from_slice(&weights.input_proj_b);
        for layer in &weights.layers {
            values.extend_from_slice(&layer.norm_weight);
            values.extend_from_slice(&layer.in_proj_w);
            values.extend_from_slice(&layer.conv1d_weight);
            values.extend_from_slice(&layer.conv1d_bias);
            values.extend_from_slice(&layer.x_proj_w);
            values.extend_from_slice(&layer.dt_proj_w);
            values.extend_from_slice(&layer.dt_proj_b);
            values.extend_from_slice(&layer.a_log);
            values.extend_from_slice(&layer.d_param);
            values.extend_from_slice(&layer.out_proj_w);
        }
        values.extend_from_slice(&weights.norm_f_weight);
        digest(&values)
    }

    let cfg = MambaConfig {
        d_model: 128,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    };
    let run = || -> (Vec<(u64, u64)>, u64) {
        let mut cpu = MambaWeights::init(&cfg, cfg.d_model, 0xDE7E_4213);
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
        for lw in cpu.layers.iter_mut() {
            lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
        }
        let session = TrainSessionCfg {
            input_dim: cfg.d_model,
            batch: 4,
            seq_len: 256,
            lr: 1e-3,
            weight_decay: 0.0,
        };
        let mut tr =
            MambaTrainer::new_full(0, &cpu, cfg, session, WeightDtype::Bf16).expect("trainer");
        tr.ctx().set_gemm_mode(GemmMode::Deterministic).unwrap();
        tr.ctx().set_bi_tensor_cores(true);
        let n = 4 * 256 * cfg.d_model;
        tr.step(&det(n, 0x11, 1.0), &det(n, 0x77, 0.1))
            .expect("warmup");
        tr.capture_graph().expect("capture");
        assert!(tr.has_graph());

        let mut replay_digests = Vec::new();
        for s in 0..2 {
            let metrics = tr
                .step(&det(n, 0x12 + s, 1.0), &det(n, 0x78 + s, 0.1))
                .expect("graph replay");
            assert!(metrics.graph_replayed);
            let stream = tr.ctx().stream.clone();
            let gradients = tr.grad_arena().to_cpu(&stream).expect("gradients");
            let master = tr.snapshot_master().expect("snapshot");
            replay_digests.push((digest(&gradients), master_digest(&master)));
        }
        let final_master = master_digest(&tr.snapshot_master().expect("snapshot"));
        (replay_digests, final_master)
    };
    let a = run();
    let b = run();
    assert_eq!(
        a, b,
        "TC graph gradient or master digest changed across runs"
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
    tile: gemm_bi_triad::TcTile,
) -> Vec<f32> {
    let (m, _k, n) = dims;
    let (qx, qw, bias) = data;
    let xt = t.typed_buf(qx, dt);
    let wt = t.typed_buf(qw, dt);
    let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
    let ops = gemm_bi_triad::TcFwdOperands {
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
    gemm_bi_triad::gemm_bi_forward_tc_with_tile(&t.ctx.stream, &t.ctx.kernels, &ops, dims, tile)
        .unwrap();
    t.ctx.stream.synchronize().unwrap();
    let mut out = vec![0.0f32; m * n];
    yt.download_f32(&t.ctx.stream, &mut out).unwrap();
    out
}

/// Launch-reality check: the TC64 kernels must have been
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
        ("gemm_bi_nn_tc64", &k.gemm_bi_nn_tc64_typed),
        ("gemm_bi_tn_tc64", &k.gemm_bi_tn_tc64_typed),
        ("gemm_bi_nt_tc64", &k.gemm_bi_nt_tc64_typed),
    ];
    for (name, kern) in tc64 {
        for (dt, f) in [("bf16", &kern.bf16), ("f16", &kern.f16)] {
            let mt = f.max_threads_per_block().unwrap();
            assert_eq!(mt, 128, "{name}_{dt}: MAX_THREADS_PER_BLOCK {mt} != 128");
        }
    }
    let tc128 = [
        ("gemm_bi_nn_tc", &k.gemm_bi_nn_tc_typed),
        ("gemm_bi_tn_tc", &k.gemm_bi_tn_tc_typed),
        ("gemm_bi_nt_tc", &k.gemm_bi_nt_tc_typed),
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
    use gemm_bi_triad::TcTile;
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
                gemm_bi_triad::gemm_bi_backward_dw_tc_with_tile(
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
                gemm_bi_triad::gemm_bi_backward_dx_tc_with_tile(
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

#[test]
fn tn_rect128x64_forced_bits_match_square_ladder() {
    use gemm_bi_triad::TcTile;

    let t = Ctx::new();
    let dims = (65usize, 129usize, 65usize);
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        let x = t.typed_buf(&quantize(&det(dims.0 * dims.1, 0x1286, 0.5), dt), dt);
        let dy = t.typed_buf(&quantize(&det(dims.0 * dims.2, 0x6403, 0.25), dt), dt);
        let initial = det(dims.1 * dims.2, 0x3201, 0.125);
        let mut results = Vec::new();
        for tile in [TcTile::Tile64, TcTile::Tile128, TcTile::Rect128x64] {
            let dw = t.f32_buf(&initial);
            gemm_bi_triad::gemm_bi_backward_dw_tc_with_tile(
                &t.ctx.stream,
                &t.ctx.kernels,
                dw.cached_ptr(),
                TypedPtr {
                    ptr: dy.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: x.cached_ptr(),
                    dtype: dt,
                },
                dims,
                tile,
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            results.push(
                dw.to_cpu(&t.ctx.stream)
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(results[0], results[1], "{dt:?} Tile64/Tile128 drift");
        assert_eq!(results[0], results[2], "{dt:?} Tile64/Rect128x64 drift");
    }
}

#[test]
fn tc64_backward_tail_contract_is_exact() {
    use gemm_bi_triad::TcTile;

    let t = Ctx::new();
    let axes = [1usize, 7, 8, 15, 16, 31, 32, 40, 48, 49, 63, 64, 65];
    let reductions = [1usize, 17, 63, 64, 65, 100, 129];

    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for (case, &axis) in axes.iter().enumerate() {
            for (rows, cols) in [(axis, 65usize), (65usize, axis)] {
                let reduction = reductions[case % reductions.len()];

                let x = t.typed_buf(
                    &quantize(&det(reduction * rows, 0x1100 + case as u32, 0.5), dt),
                    dt,
                );
                let dy = t.typed_buf(
                    &quantize(&det(reduction * cols, 0x2200 + case as u32, 0.25), dt),
                    dt,
                );
                let initial = det(rows * cols, 0x3300 + case as u32, 0.125);
                let mut dw_results = Vec::new();
                for tile in [TcTile::Tile64, TcTile::Tile128] {
                    let dw = t.f32_buf(&initial);
                    gemm_bi_triad::gemm_bi_backward_dw_tc_with_tile(
                        &t.ctx.stream,
                        &t.ctx.kernels,
                        dw.cached_ptr(),
                        TypedPtr {
                            ptr: dy.cached_ptr(),
                            dtype: dt,
                        },
                        TypedPtr {
                            ptr: x.cached_ptr(),
                            dtype: dt,
                        },
                        (reduction, rows, cols),
                        tile,
                    )
                    .unwrap();
                    t.ctx.stream.synchronize().unwrap();
                    dw_results.push(dw.to_cpu(&t.ctx.stream).unwrap());
                }
                assert_eq!(
                    dw_results[0]
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    dw_results[1]
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "{dt:?} TN Tile64/Tile128 mismatch at M{reduction} K{rows} N{cols}"
                );
                assert!(
                    dw_results[0]
                        .iter()
                        .zip(&initial)
                        .any(|(after, before)| after.to_bits() != before.to_bits()),
                    "{dt:?} TN did not accumulate at M{reduction} K{rows} N{cols}"
                );
                let dw_zero = GpuBuffer::zeros(&t.ctx.stream, rows * cols).unwrap();
                gemm_bi_triad::gemm_bi_backward_dw_tc_with_tile(
                    &t.ctx.stream,
                    &t.ctx.kernels,
                    dw_zero.cached_ptr(),
                    TypedPtr {
                        ptr: dy.cached_ptr(),
                        dtype: dt,
                    },
                    TypedPtr {
                        ptr: x.cached_ptr(),
                        dtype: dt,
                    },
                    (reduction, rows, cols),
                    TcTile::Tile64,
                )
                .unwrap();
                t.ctx.stream.synchronize().unwrap();
                let product = dw_zero.to_cpu(&t.ctx.stream).unwrap();
                assert_eq!(
                    dw_results[0]
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    product
                        .iter()
                        .zip(&initial)
                        .map(|(value, old)| (value + old).to_bits())
                        .collect::<Vec<_>>(),
                    "{dt:?} TN did not preserve the incoming dW at M{reduction} K{rows} N{cols}"
                );

                let nt_dy = t.typed_buf(
                    &quantize(&det(rows * reduction, 0x4400 + case as u32, 0.25), dt),
                    dt,
                );
                let w = t.typed_buf(
                    &quantize(&det(cols * reduction, 0x5500 + case as u32, 0.5), dt),
                    dt,
                );
                let sentinel = quantize(&vec![3.25f32; rows * cols], dt);
                let mut dx_results = Vec::new();
                for tile in [TcTile::Tile64, TcTile::Tile128] {
                    let dx = t.typed_buf(&sentinel, dt);
                    gemm_bi_triad::gemm_bi_backward_dx_tc_with_tile(
                        &t.ctx.stream,
                        &t.ctx.kernels,
                        TypedPtr {
                            ptr: dx.cached_ptr(),
                            dtype: dt,
                        },
                        TypedPtr {
                            ptr: nt_dy.cached_ptr(),
                            dtype: dt,
                        },
                        TypedPtr {
                            ptr: w.cached_ptr(),
                            dtype: dt,
                        },
                        (rows, cols, reduction),
                        tile,
                    )
                    .unwrap();
                    t.ctx.stream.synchronize().unwrap();
                    let mut result = vec![0.0f32; rows * cols];
                    dx.download_f32(&t.ctx.stream, &mut result).unwrap();
                    dx_results.push(result);
                }
                assert_eq!(
                    dx_results[0]
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    dx_results[1]
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "{dt:?} NT Tile64/Tile128 mismatch at M{rows} K{cols} N{reduction}"
                );
                assert!(
                    dx_results[0]
                        .iter()
                        .zip(&sentinel)
                        .any(|(after, before)| after.to_bits() != before.to_bits()),
                    "{dt:?} NT did not overwrite at M{rows} K{cols} N{reduction}"
                );
                let dx_zero = DtypedBuf::zeros(&t.ctx.stream, rows * cols, dt).unwrap();
                gemm_bi_triad::gemm_bi_backward_dx_tc_with_tile(
                    &t.ctx.stream,
                    &t.ctx.kernels,
                    TypedPtr {
                        ptr: dx_zero.cached_ptr(),
                        dtype: dt,
                    },
                    TypedPtr {
                        ptr: nt_dy.cached_ptr(),
                        dtype: dt,
                    },
                    TypedPtr {
                        ptr: w.cached_ptr(),
                        dtype: dt,
                    },
                    (rows, cols, reduction),
                    TcTile::Tile64,
                )
                .unwrap();
                t.ctx.stream.synchronize().unwrap();
                let mut from_zero = vec![0.0f32; rows * cols];
                dx_zero.download_f32(&t.ctx.stream, &mut from_zero).unwrap();
                assert_eq!(
                    dx_results[0]
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    from_zero
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "{dt:?} NT retained the incoming output at M{rows} K{cols} N{reduction}"
                );
            }
        }
    }
}

#[test]
fn tc64_backward_tail_repeats_are_bit_stable() {
    use gemm_bi_triad::TcTile;

    let t = Ctx::new();
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        let (m, k, n) = (129usize, 49usize, 65usize);
        let x = t.typed_buf(&quantize(&det(m * k, 0x6100, 0.5), dt), dt);
        let dy = t.typed_buf(&quantize(&det(m * n, 0x6200, 0.25), dt), dt);
        let initial = det(k * n, 0x6300, 0.125);
        let mut reference = None;
        for _ in 0..20 {
            let dw = t.f32_buf(&initial);
            gemm_bi_triad::gemm_bi_backward_dw_tc_with_tile(
                &t.ctx.stream,
                &t.ctx.kernels,
                dw.cached_ptr(),
                TypedPtr {
                    ptr: dy.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: x.cached_ptr(),
                    dtype: dt,
                },
                (m, k, n),
                TcTile::Tile64,
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let bits = dw
                .to_cpu(&t.ctx.stream)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            if let Some(expected) = &reference {
                assert_eq!(
                    &bits, expected,
                    "{dt:?} TN changed across repeated launches"
                );
            } else {
                reference = Some(bits);
            }
        }

        let nt_dy = t.typed_buf(&quantize(&det(m * n, 0x6400, 0.25), dt), dt);
        let w = t.typed_buf(&quantize(&det(k * n, 0x6500, 0.5), dt), dt);
        let sentinel = quantize(&vec![-2.5f32; m * k], dt);
        let mut reference = None;
        for _ in 0..20 {
            let dx = t.typed_buf(&sentinel, dt);
            gemm_bi_triad::gemm_bi_backward_dx_tc_with_tile(
                &t.ctx.stream,
                &t.ctx.kernels,
                TypedPtr {
                    ptr: dx.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: nt_dy.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: w.cached_ptr(),
                    dtype: dt,
                },
                (m, k, n),
                TcTile::Tile64,
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let mut result = vec![0.0f32; m * k];
            dx.download_f32(&t.ctx.stream, &mut result).unwrap();
            let bits = result.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            if let Some(expected) = &reference {
                assert_eq!(
                    &bits, expected,
                    "{dt:?} NT changed across repeated launches"
                );
            } else {
                reference = Some(bits);
            }
        }
    }
}

#[test]
fn tc64_backward_qualified_routes_match_the_forced_kernel() {
    use gemm_bi_triad::TcTile;

    let t = Ctx::new();
    let tn_shapes = [
        (1024usize, 8usize, 256usize),
        (2048, 16, 512),
        (2048, 48, 1536),
        (1024, 256, 40),
        (2048, 512, 48),
    ];
    let nt_shapes = [
        (1024usize, 8usize, 256usize),
        (2048, 16, 512),
        (2048, 48, 1536),
        (32, 256, 1024),
    ];

    t.ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    t.ctx.set_bi_tensor_cores(true);
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for (case, &(m, k, n)) in tn_shapes.iter().enumerate() {
            let x = t.typed_buf(&quantize(&det(m * k, 0x7100 + case as u32, 0.5), dt), dt);
            let dy = t.typed_buf(&quantize(&det(m * n, 0x7200 + case as u32, 0.25), dt), dt);
            let initial = det(k * n, 0x7300 + case as u32, 0.125);
            let forced = t.f32_buf(&initial);
            let automatic = t.f32_buf(&initial);
            let resolved = t.f32_buf(&initial);
            let dyp = TypedPtr {
                ptr: dy.cached_ptr(),
                dtype: dt,
            };
            let xp = TypedPtr {
                ptr: x.cached_ptr(),
                dtype: dt,
            };
            gemm_bi_triad::gemm_bi_backward_dw_tc_with_tile(
                &t.ctx.stream,
                &t.ctx.kernels,
                forced.cached_ptr(),
                dyp,
                xp,
                (m, k, n),
                TcTile::Tile64,
            )
            .unwrap();
            let tile = gemm_bi_triad::gemm_bi_backward_dw_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                automatic.cached_ptr(),
                dyp,
                xp,
                (m, k, n),
            )
            .unwrap();
            assert_eq!(tile, TcTile::Tile64, "{dt:?} TN route at M{m} K{k} N{n}");
            mamba_rs::mamba_ssm::gpu::blas::gemm_bi_backward_dw_typed(
                &t.ctx,
                resolved.cached_ptr(),
                dyp,
                xp,
                (m, k, n),
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let forced_bits = forced
                .to_cpu(&t.ctx.stream)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            assert_eq!(
                forced_bits,
                automatic
                    .to_cpu(&t.ctx.stream)
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                "{dt:?} TN automatic route differs from forced Tile64 at M{m} K{k} N{n}"
            );
            assert_eq!(
                forced_bits,
                resolved
                    .to_cpu(&t.ctx.stream)
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                "{dt:?} TN resolved path differs from forced Tile64 at M{m} K{k} N{n}"
            );
        }

        for (case, &(m, k, n)) in nt_shapes.iter().enumerate() {
            let dy = t.typed_buf(&quantize(&det(m * n, 0x7400 + case as u32, 0.25), dt), dt);
            let w = t.typed_buf(&quantize(&det(k * n, 0x7500 + case as u32, 0.5), dt), dt);
            let sentinel = quantize(&vec![1.75f32; m * k], dt);
            let forced = t.typed_buf(&sentinel, dt);
            let automatic = t.typed_buf(&sentinel, dt);
            let resolved = t.typed_buf(&sentinel, dt);
            let dyp = TypedPtr {
                ptr: dy.cached_ptr(),
                dtype: dt,
            };
            let wp = TypedPtr {
                ptr: w.cached_ptr(),
                dtype: dt,
            };
            gemm_bi_triad::gemm_bi_backward_dx_tc_with_tile(
                &t.ctx.stream,
                &t.ctx.kernels,
                TypedPtr {
                    ptr: forced.cached_ptr(),
                    dtype: dt,
                },
                dyp,
                wp,
                (m, k, n),
                TcTile::Tile64,
            )
            .unwrap();
            let tile = gemm_bi_triad::gemm_bi_backward_dx_tc(
                &t.ctx.stream,
                &t.ctx.kernels,
                TypedPtr {
                    ptr: automatic.cached_ptr(),
                    dtype: dt,
                },
                dyp,
                wp,
                (m, k, n),
            )
            .unwrap();
            assert_eq!(tile, TcTile::Tile64, "{dt:?} NT route at M{m} K{k} N{n}");
            mamba_rs::mamba_ssm::gpu::blas::gemm_bi_backward_dx_typed(
                &t.ctx,
                TypedPtr {
                    ptr: resolved.cached_ptr(),
                    dtype: dt,
                },
                dyp,
                wp,
                (m, k, n),
            )
            .unwrap();
            t.ctx.stream.synchronize().unwrap();
            let mut forced_host = vec![0.0f32; m * k];
            let mut automatic_host = vec![0.0f32; m * k];
            let mut resolved_host = vec![0.0f32; m * k];
            forced
                .download_f32(&t.ctx.stream, &mut forced_host)
                .unwrap();
            automatic
                .download_f32(&t.ctx.stream, &mut automatic_host)
                .unwrap();
            resolved
                .download_f32(&t.ctx.stream, &mut resolved_host)
                .unwrap();
            let forced_bits = forced_host
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            assert_eq!(
                forced_bits,
                automatic_host
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                "{dt:?} NT automatic route differs from forced Tile64 at M{m} K{k} N{n}"
            );
            assert_eq!(
                forced_bits,
                resolved_host
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                "{dt:?} NT resolved path differs from forced Tile64 at M{m} K{k} N{n}"
            );
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
            gemm_bi_triad::gemm_bi_forward(
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
            gemm_bi_triad::gemm_bi_backward_dw(
                &t.ctx.stream,
                &t.ctx.kernels,
                dw_ref.cached_ptr(),
                &dy32,
                &x32,
                (m, k, n),
            )
            .unwrap();
            let mut dx_ref = GpuBuffer::zeros(&t.ctx.stream, m * k).unwrap();
            gemm_bi_triad::gemm_bi_backward_dx(
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
                gemm_bi_triad::gemm_bi_backward_dw_tc(
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
            gemm_bi_triad::gemm_bi_backward_dx_tc(
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
    use gemm_bi_triad::TcTile;
    let t = Ctx::new();
    let dt = WeightDtype::Bf16;
    let k = 64usize;
    let compute_capability = t
        .ctx
        .stream
        .context()
        .compute_capability()
        .expect("CUDA compute capability");
    let multiprocessor_count = t
        .ctx
        .stream
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .unwrap();
    let multiprocessor_count = usize::try_from(multiprocessor_count).unwrap();
    let square_tile = |tiles128: usize| {
        if tiles128 * 2 >= multiprocessor_count {
            TcTile::Tile128
        } else {
            TcTile::Tile64
        }
    };
    let underfilled_tail = |tiles64: usize| {
        if tiles64 * 16 < multiprocessor_count * 3 {
            TcTile::Thin16
        } else {
            TcTile::Tile64
        }
    };
    // Cross both the base half-wave threshold and the TN rectangular
    // one-wave guard on the device running this contract test.
    let n_wide = 128 * (multiprocessor_count * 9 / 8 + 1);

    let route = |m: usize, n: usize| -> Result<TcTile, String> {
        let qx = quantize(&det(m * k, 11, 1.0), dt);
        let qw = quantize(&det(k * n, 22, 0.5), dt);
        let xt = t.typed_buf(&qx, dt);
        let wt = t.typed_buf(&qw, dt);
        let yt = DtypedBuf::zeros(&t.ctx.stream, m * n, dt).unwrap();
        let r = gemm_bi_triad::gemm_bi_forward_tc(
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

    // Below the Thin16 column floor -> honest UNCOVERED (blas.rs matvec
    // fallback). Since the ladder unification, N is the only uncovered axis.
    for (m, n) in [(1usize, 31usize), (63, 31), (4096, 31)] {
        let err = route(m, n).unwrap_err();
        assert!(
            err.starts_with("UNCOVERED"),
            "M{m} N{n}: expected UNCOVERED prefix, got: {err}"
        );
    }
    // The Thin16 rung: every M at N in [32, 64), and M <= 64 at any
    // covered N (the measured Thin16/Tile64 crossover, bit-free pick).
    assert_eq!(route(63, 64).unwrap(), TcTile::Thin16);
    assert_eq!(route(64, 63).unwrap(), TcTile::Thin16);
    assert_eq!(route(63, 4096).unwrap(), TcTile::Thin16);
    assert_eq!(route(4096, 63).unwrap(), TcTile::Thin16);
    assert_eq!(route(64, 64).unwrap(), TcTile::Thin16);
    assert_eq!(route(1, 32).unwrap(), TcTile::Thin16);
    // The measured short-reduction tail uses Thin16 only while its Tile64
    // grid remains below the device-aware wave floor.
    assert_eq!(route(65, 65).unwrap(), underfilled_tail(4));
    assert_eq!(route(127, 127).unwrap(), underfilled_tail(4));
    assert_eq!(route(127, 4096).unwrap(), TcTile::Tile64);
    assert_eq!(route(4096, 127).unwrap(), underfilled_tail(128));
    // Square routes cross from Tile64 to Tile128 at a half-device wave.
    assert_eq!(route(128, 128).unwrap(), square_tile(1));
    assert_eq!(route(1024, 512).unwrap(), square_tile(32));
    assert_eq!(route(128, n_wide).unwrap(), TcTile::Tile128);
    assert_eq!(route(2048, 3072).unwrap(), square_tile(384));

    // dW routes on (K_out, N), never on the reduction dim M.
    let dw_route = |kk: usize, n: usize| -> Result<TcTile, String> {
        let m = 256usize;
        let qx = quantize(&det(m * kk, 44, 1.0), dt);
        let qdy = quantize(&det(m * n, 55, 0.5), dt);
        let xt = t.typed_buf(&qx, dt);
        let dyt = t.typed_buf(&qdy, dt);
        let dw = GpuBuffer::zeros(&t.ctx.stream, kk * n).unwrap();
        let r = gemm_bi_triad::gemm_bi_backward_dw_tc(
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
    assert_eq!(dw_route(63, 512).unwrap(), TcTile::Tile64);
    assert!(dw_route(63, 63).unwrap_err().starts_with("UNCOVERED"));
    assert_eq!(dw_route(64, 64).unwrap(), TcTile::Tile64);
    assert_eq!(dw_route(128, 512).unwrap(), TcTile::Tile64); // d128 in_proj dW
    // The measured SM89 policy keeps long-reduction dW on Tile64 even when
    // the portable square-grid policy has enough work for Tile128.
    let wide_dw_tile = if compute_capability == (8, 9) {
        TcTile::Tile64
    } else {
        TcTile::Tile128
    };
    assert_eq!(dw_route(128, n_wide).unwrap(), wide_dw_tile);

    // dX routes on (M, K_out).
    let dx_route = |m: usize, kk: usize| -> Result<TcTile, String> {
        let n = 256usize;
        let qdy = quantize(&det(m * n, 55, 0.5), dt);
        let qw = quantize(&det(kk * n, 77, 0.5), dt);
        let dyt = t.typed_buf(&qdy, dt);
        let wt = t.typed_buf(&qw, dt);
        let dxt = DtypedBuf::zeros(&t.ctx.stream, m * kk, dt).unwrap();
        let r = gemm_bi_triad::gemm_bi_backward_dx_tc(
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
    assert_eq!(dx_route(63, 512).unwrap(), TcTile::Tile64);
    assert!(dx_route(63, 63).unwrap_err().starts_with("UNCOVERED"));
    assert_eq!(dx_route(64, 64).unwrap(), TcTile::Tile64);
    assert_eq!(dx_route(1024, 128).unwrap(), square_tile(8));
    assert_eq!(dx_route(9216, 128).unwrap(), square_tile(72));
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

/// The portable module composes the tc64 TN stream-K fragment on every
/// sm80-family board except CC 12.x, whose boards run the SM120 stream-K
/// kernel; the forced stream-K tests have nothing to launch there.
fn portable_module_composes_streamk() -> bool {
    let device = GpuDevice::new(0).expect("device 0");
    if device.compute_capability.0 == 12 {
        eprintln!(
            "skip: the portable stream-K fragment is not composed for CC {:?}",
            device.compute_capability
        );
        return false;
    }
    true
}

/// The stream-K dW schedule folds per-CTA partial slabs in a fixed order, so
/// its bits differ from the tiled ladder on a multi-CTA grid; the contract it
/// carries is bit-stability per device plus agreement with the reference to
/// f32 accumulation tolerance. With a one-CTA grid every tile is one segment
/// in ascending order and the output equals the tiled kernel bit for bit.
#[test]
fn tn_tc64_streamk_forced_agrees_with_reference_and_repeats_bit_for_bit() {
    use gemm_bi_triad::TcTile;

    if !portable_module_composes_streamk() {
        return;
    }
    let t = Ctx::new();
    let shapes: [(usize, usize, usize); 7] = [
        (10400, 384, 384),
        (4621, 200, 136),
        (2048, 48, 1536),
        (65, 129, 65),
        (200, 130, 70),
        (1, 64, 64),
        (8192, 128, 128),
    ];
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for (index, dims) in shapes.into_iter().enumerate() {
            let seed = 0x5000 + index as u32 * 7;
            let qx = quantize(&det(dims.0 * dims.1, seed, 0.5), dt);
            let qdy = quantize(&det(dims.0 * dims.2, seed + 1, 0.25), dt);
            let initial = det(dims.1 * dims.2, seed + 2, 0.125);
            let x = t.typed_buf(&qx, dt);
            let dy = t.typed_buf(&qdy, dt);
            let launch = |tile: TcTile| {
                let dw = t.f32_buf(&initial);
                gemm_bi_triad::gemm_bi_backward_dw_tc_with_tile(
                    &t.ctx.stream,
                    &t.ctx.kernels,
                    dw.cached_ptr(),
                    TypedPtr {
                        ptr: dy.cached_ptr(),
                        dtype: dt,
                    },
                    TypedPtr {
                        ptr: x.cached_ptr(),
                        dtype: dt,
                    },
                    dims,
                    tile,
                )
                .unwrap();
                t.ctx.stream.synchronize().unwrap();
                dw.to_cpu(&t.ctx.stream).unwrap()
            };
            let first = launch(TcTile::Tile64StreamK);
            let second = launch(TcTile::Tile64StreamK);
            assert_eq!(
                first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                second.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{dt:?} {dims:?}: stream-K repeat drift"
            );
            let tiled = launch(TcTile::Tile64);
            // Reference: dW[k][n] = initial + sum_m x[m][k] * dy[m][n] in f64.
            let (m, k, n) = dims;
            let mut worst = 0.0f64;
            for kk in 0..k {
                for nn in 0..n {
                    let mut sum = 0.0f64;
                    for mm in 0..m {
                        sum += f64::from(qx[mm * k + kk]) * f64::from(qdy[mm * n + nn]);
                    }
                    let expected = f64::from(initial[kk * n + nn]) + sum;
                    let got = f64::from(first[kk * n + nn]);
                    let scale = expected
                        .abs()
                        .max(f64::from(tiled[kk * n + nn]).abs())
                        .max(1.0);
                    worst = worst.max((got - expected).abs() / scale);
                }
            }
            // f32 accumulation over m products, folded in a different order.
            let bound = 4e-6 * (m as f64).sqrt().max(1.0);
            assert!(
                worst <= bound,
                "{dt:?} {dims:?}: stream-K differs from the f64 reference by {worst:e} (bound {bound:e})"
            );
        }
    }
}

#[test]
fn tn_tc64_streamk_qualifies_under_its_own_contract_on_a_persistent_grid() {
    use gemm_bi_triad::{
        PhysicalQualificationRequest, PhysicalQualificationRoute, TcTile, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ResolvedGemmOp, ResolvedNumericContract, ResolvedOutputOwnership,
    };

    if !portable_module_composes_streamk() {
        return;
    }
    let t = Ctx::new();
    let dims = (2048usize, 384usize, 384usize);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let physical = qualify_physical_launch(
            &t.ctx,
            PhysicalQualificationRequest::contiguous(
                ResolvedGemmOp::Tn,
                dims,
                PhysicalQualificationRoute::HalfForced {
                    dtype,
                    tile: TcTile::Tile64StreamK,
                },
            ),
        )
        .expect("qualify the forced stream-K dW route");
        let evidence = physical.evidence();
        assert!(
            evidence.eager_graph_equal(),
            "{dtype:?}: eager and graph replay differ"
        );
        assert_eq!(evidence.launch_count(), 1, "{dtype:?}");
        let [node] = evidence.nodes() else {
            panic!(
                "{dtype:?}: forced stream-K dW recorded {} nodes",
                evidence.nodes().len()
            );
        };
        assert!(
            node.symbol.starts_with("gemm_bi_tn_tc64_streamk"),
            "{dtype:?}: {}",
            node.symbol
        );
        assert_eq!(
            node.numeric_contract,
            Some(ResolvedNumericContract::MmaSyncF32StreamKFixedOrderV1),
            "{dtype:?}"
        );
        assert_eq!(
            node.ownership,
            Some(ResolvedOutputOwnership::OwnerCtaPerOutputTileStreamKFixedOrderV1),
            "{dtype:?}"
        );
        // 36 tiles by 32 slabs: a persistent grid of one CTA per
        // multiprocessor, well under the 1152 units it walks.
        assert!(
            node.launch.grid_dim.0 > 1 && node.launch.grid_dim.0 < 1152,
            "{dtype:?}: grid {:?}",
            node.launch.grid_dim
        );
        assert_eq!(node.launch.block_dim, (128, 1, 1), "{dtype:?}");
    }
    // The lease restored the context's own policy, the stream-K default of
    // the tensor-core tier.
    assert_eq!(
        t.ctx.half_triad_policy(),
        mamba_rs::mamba_ssm::gpu::context::HalfTriadPolicy::AllowStreamKFixedOrderV1
    );
}

/// On SM89 the automatic dW route takes the stream-K schedule for an
/// underfilled deep batch cell only under the permitting half policy. The
/// native half path records no eager route, so the route is proven by its
/// bits: the automatic launch equals the forced kernel of the schedule the
/// policy names, bit for bit, and the two schedules are distinguishable.
#[test]
fn sm89_automatic_dw_takes_stream_k_only_under_the_half_policy() {
    use gemm_bi_triad::TcTile;
    use mamba_rs::mamba_ssm::gpu::blas::gemm_bi_backward_dw_typed;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, HalfTriadPolicy};

    let device = GpuDevice::new(0).expect("device 0");
    if device.compute_capability != (8, 9) {
        eprintln!(
            "skip: the stream-K dW rule is measured on CC 8.9, this board is {:?}",
            device.compute_capability
        );
        return;
    }
    let t = Ctx::new();
    t.ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    t.ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    t.ctx.set_bi_tensor_cores(true);
    let dims = (10400usize, 384usize, 384usize);
    let dt = WeightDtype::Bf16;
    let x = t.typed_buf(&quantize(&det(dims.0 * dims.1, 0x91, 0.5), dt), dt);
    let dy = t.typed_buf(&quantize(&det(dims.0 * dims.2, 0x92, 0.25), dt), dt);
    let dyp = TypedPtr {
        ptr: dy.cached_ptr(),
        dtype: dt,
    };
    let xp = TypedPtr {
        ptr: x.cached_ptr(),
        dtype: dt,
    };
    let bits = |dw: &GpuBuffer| {
        t.ctx.stream.synchronize().unwrap();
        dw.to_cpu(&t.ctx.stream)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    };
    let automatic = |policy: HalfTriadPolicy| {
        t.ctx.set_half_triad_policy(policy);
        let dw = GpuBuffer::zeros(&t.ctx.stream, dims.1 * dims.2).unwrap();
        gemm_bi_backward_dw_typed(&t.ctx, dw.cached_ptr(), dyp, xp, dims).expect("automatic dW");
        bits(&dw)
    };
    let forced = |tile: TcTile| {
        let dw = GpuBuffer::zeros(&t.ctx.stream, dims.1 * dims.2).unwrap();
        gemm_bi_triad::gemm_bi_backward_dw_tc_with_tile(
            &t.ctx.stream,
            &t.ctx.kernels,
            dw.cached_ptr(),
            dyp,
            xp,
            dims,
            tile,
        )
        .unwrap();
        bits(&dw)
    };
    let tiled = forced(TcTile::Tile64);
    let stream_k = forced(TcTile::Tile64StreamK);
    assert_ne!(
        tiled, stream_k,
        "the two schedules must be distinguishable on this shape for the route proof"
    );
    assert_eq!(
        automatic(HalfTriadPolicy::TiledParityV1),
        tiled,
        "the default policy must launch the tiled tc64 kernel"
    );
    assert_eq!(
        automatic(HalfTriadPolicy::AllowStreamKFixedOrderV1),
        stream_k,
        "the permitting policy must launch the stream-K kernel"
    );
    t.ctx.set_half_triad_policy(HalfTriadPolicy::TiledParityV1);
    assert_eq!(automatic(HalfTriadPolicy::TiledParityV1), tiled);
    // The stream-K fold differs in order, not in value: accumulation tolerance.
    let worst = tiled
        .iter()
        .zip(&stream_k)
        .map(|(a, b)| {
            let a = f64::from(f32::from_bits(*a));
            let b = f64::from(f32::from_bits(*b));
            (a - b).abs() / a.abs().max(1.0)
        })
        .fold(0.0f64, f64::max);
    let bound = 4e-6 * (dims.0 as f64).sqrt();
    assert!(
        worst <= bound,
        "stream-K differs from tiled by {worst:e} (bound {bound:e})"
    );
}
