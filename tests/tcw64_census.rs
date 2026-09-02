//! On-box census for the fragment-reuse rung: per-element byte identity
//! with the 128-tile, resource sanity, and the event-timed comparison
//! at the fat training shapes. The rung enters the dispatch ladder only
//! on a green identity census plus a measured win.
#![cfg(feature = "cuda")]

mod common;

use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags;
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let base = ((s & 0xFFFFFF) as f32 / 16777216.0) * 0.5 - 0.25;
            match i % 5 {
                0 => 1024.0,
                1 => -1024.0,
                _ => base,
            }
        })
        .collect()
}

/// One rung's launch geometry: output tile, dynamic smem, block size.
struct Rung<'a> {
    kern: &'a cudarc::driver::CudaFunction,
    tile: (usize, usize),
    smem: u32,
    threads: u32,
}

fn launch_tile(
    ctx: &GpuCtx,
    rung: &Rung<'_>,
    c: &DtypedBuf,
    a: &DtypedBuf,
    b: &DtypedBuf,
    bias: u64,
    dims: (usize, usize, usize),
) {
    let Rung {
        kern,
        tile,
        smem,
        threads,
    } = *rung;
    let (m, k, n) = dims;
    let grid = (m.div_ceil(tile.0) * n.div_ceil(tile.1)) as u32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem,
    };
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let (mi, ni, ki) = (m as i32, n as i32, k as i32);
    let cp = c.cached_ptr();
    let ap = a.cached_ptr();
    let bp = b.cached_ptr();
    let mut bld = ctx.stream.launch_builder(kern);
    bld.arg(&cp);
    bld.arg(&ap);
    bld.arg(&bp);
    bld.arg(&bias);
    bld.arg(&alpha);
    bld.arg(&beta);
    bld.arg(&mi);
    bld.arg(&ni);
    bld.arg(&ki);
    bld.arg(&ki);
    bld.arg(&ni);
    bld.arg(&ni);
    unsafe { bld.launch(cfg) }.expect("tile launch");
}

/// Byte identity: the fragment-reuse rung against the 128-tile, over
/// tails on every axis, both dtypes, bias on and off.
#[test]
#[ignore = "needs a CUDA device"]
fn tcw64_bit_identical_to_tc128() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let st = &ctx.stream;

    let shapes = [
        (128usize, 384usize, 384usize),
        (2048, 768, 3072),
        (515, 768, 640),
        (2048, 100, 1024),
        (4621, 384, 1928),
        (129, 65, 257),
    ];
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for (m, k, n) in shapes {
            for with_bias in [false, true] {
                let a = DtypedBuf::zeros(st, m * k, dt).unwrap();
                a.upload_f32(st, &det(m * k, 7)).unwrap();
                let b = DtypedBuf::zeros(st, k * n, dt).unwrap();
                b.upload_f32(st, &det(k * n, 9)).unwrap();
                let bias_buf = GpuBuffer::from_cpu(st, &det(n, 11)).unwrap();
                let bias = if with_bias { bias_buf.cached_ptr() } else { 0 };
                let c_ref = DtypedBuf::zeros(st, m * n, dt).unwrap();
                let c_new = DtypedBuf::zeros(st, m * n, dt).unwrap();
                launch_tile(
                    &ctx,
                    &Rung {
                        kern: ctx.kernels.gemm_bi_nn_tc128_typed.get(dt),
                        tile: (128, 128),
                        smem: 71_680,
                        threads: 256,
                    },
                    &c_ref,
                    &a,
                    &b,
                    bias,
                    (m, k, n),
                );
                st.synchronize().unwrap();
                let mut hr = vec![0.0f32; m * n];
                c_ref.download_f32(st, &mut hr).unwrap();
                for (label, rung) in [
                    (
                        "tcw64",
                        Rung {
                            kern: ctx.kernels.gemm_bi_nn_tcw64_typed.get(dt),
                            tile: (128, 128),
                            smem: 65_536,
                            threads: 128,
                        },
                    ),
                    (
                        "tcwn64",
                        Rung {
                            kern: ctx.kernels.gemm_bi_nn_tcwn64_typed.get(dt),
                            tile: (128, 256),
                            smem: 98_304,
                            threads: 256,
                        },
                    ),
                ] {
                    c_new.upload_f32(st, &vec![0.0; m * n]).unwrap();
                    launch_tile(&ctx, &rung, &c_new, &a, &b, bias, (m, k, n));
                    st.synchronize().unwrap();
                    let mut hn = vec![0.0f32; m * n];
                    c_new.download_f32(st, &mut hn).unwrap();
                    let diff = hr
                        .iter()
                        .zip(&hn)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count();
                    assert_eq!(
                        diff, 0,
                        "{label} {dt:?} M{m} K{k} N{n} bias={with_bias}: {diff} differ"
                    );
                }
            }
        }
    }
    println!("tcw64 == tc128 bitwise on every cell");
}

/// Resource sanity: no local spill, and the register count on record.
#[test]
#[ignore = "needs a CUDA device"]
fn tcw64_resources() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    for dt in [WeightDtype::Bf16, WeightDtype::F16] {
        for (label, f) in [
            ("tcw64", ctx.kernels.gemm_bi_nn_tcw64_typed.get(dt)),
            ("tcwn64", ctx.kernels.gemm_bi_nn_tcwn64_typed.get(dt)),
        ] {
            let regs = f.num_regs().unwrap();
            let local = f.local_size_bytes().unwrap();
            println!("{label} {dt:?}: regs={regs} local={local}");
            assert_eq!(local, 0, "{label} {dt:?}: local spill of {local} bytes");
        }
    }
}

/// Promotion grid for the wide rung: every candidate point measured
/// with four alternating rounds per arm; a point is a WIN only when
/// the wide rung is at least three percent ahead in every round pair
/// and both arms are tight (p-spread inside 15 percent), a LOSS
/// symmetrically, and everything else a TIE - the promotion rule's
/// band is drawn from this table, never the other way round.
#[test]
#[ignore = "record-lane instrument (GPU, quiet card)"]
fn tcwn64_promotion_grid() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let st = &ctx.stream;
    let cuda = st.context();
    let dt = WeightDtype::Bf16;
    let mk_ev = || {
        cuda.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event")
    };
    let ms_grid = [2048usize, 3072, 4096, 6144];
    let ks_grid = [768usize, 1536];
    let ns_grid = [1536usize, 2304, 3072];
    let mut wins = Vec::new();
    for m in ms_grid {
        for k in ks_grid {
            for n in ns_grid {
                let a = DtypedBuf::zeros(st, m * k, dt).unwrap();
                a.upload_f32(st, &det(m * k, 7)).unwrap();
                let b = DtypedBuf::zeros(st, k * n, dt).unwrap();
                b.upload_f32(st, &det(k * n, 9)).unwrap();
                let c = DtypedBuf::zeros(st, m * n, dt).unwrap();
                let one = |rung: &Rung<'_>| -> f64 {
                    for _ in 0..8 {
                        launch_tile(&ctx, rung, &c, &a, &b, 0, (m, k, n));
                    }
                    st.synchronize().unwrap();
                    let s_ev = mk_ev();
                    let e_ev = mk_ev();
                    s_ev.record(st).unwrap();
                    for _ in 0..25 {
                        launch_tile(&ctx, rung, &c, &a, &b, 0, (m, k, n));
                    }
                    e_ev.record(st).unwrap();
                    st.synchronize().unwrap();
                    f64::from(s_ev.elapsed_ms(&e_ev).unwrap()) * 1000.0 / 25.0
                };
                let r128 = Rung {
                    kern: ctx.kernels.gemm_bi_nn_tc128_typed.get(dt),
                    tile: (128, 128),
                    smem: 71_680,
                    threads: 256,
                };
                let rwn = Rung {
                    kern: ctx.kernels.gemm_bi_nn_tcwn64_typed.get(dt),
                    tile: (128, 256),
                    smem: 98_304,
                    threads: 256,
                };
                let mut t128 = Vec::new();
                let mut twn = Vec::new();
                for round in 0..4 {
                    if round % 2 == 0 {
                        t128.push(one(&r128));
                        twn.push(one(&rwn));
                    } else {
                        twn.push(one(&rwn));
                        t128.push(one(&r128));
                    }
                }
                let spread = |v: &[f64]| {
                    v.iter().cloned().fold(0.0f64, f64::max)
                        / v.iter().cloned().fold(f64::MAX, f64::min)
                };
                let tight = spread(&t128) <= 1.15 && spread(&twn) <= 1.15;
                let all_win = t128.iter().zip(&twn).all(|(a, b)| (a - b) / a >= 0.03);
                let all_loss = t128.iter().zip(&twn).all(|(a, b)| (b - a) / a >= 0.03);
                let p50 = |v: &[f64]| {
                    let mut s = v.to_vec();
                    s.sort_by(f64::total_cmp);
                    s[s.len() / 2]
                };
                let verdict = if !tight {
                    "UNSTABLE"
                } else if all_win {
                    "WIN"
                } else if all_loss {
                    "LOSS"
                } else {
                    "tie"
                };
                println!(
                    "M{m:<5} K{k:<5} N{n:<5}: tc128 {:7.1}us  tcwn64 {:7.1}us  {:+5.1}%  {verdict}",
                    p50(&t128),
                    p50(&twn),
                    (p50(&t128) / p50(&twn) - 1.0) * 100.0
                );
                common::evidence::record(
                    "tcw64_census",
                    "promotion",
                    &format!("M{m}K{k}N{n}"),
                    verdict,
                )
                .expect("acceptance evidence");
                if verdict == "WIN" {
                    wins.push((m, k, n));
                }
            }
        }
    }
    println!("WIN points: {wins:?}");
}

/// Event-timed comparison at the fat shapes, alternating groups.
#[test]
#[ignore = "record-lane instrument (GPU, quiet card)"]
fn tcw64_vs_tc128_fat_shapes() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let st = &ctx.stream;
    let cuda = st.context();
    let dt = WeightDtype::Bf16;
    let mk_ev = || {
        cuda.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event")
    };
    let shapes = [
        (2048usize, 768usize, 3072usize),
        (4096, 1536, 3072),
        (4096, 768, 3072),
        (2048, 1536, 1536),
        (2048, 2304, 768),
        (2048, 768, 2304),
        (4621, 384, 1928),
    ];
    for (m, k, n) in shapes {
        let a = DtypedBuf::zeros(st, m * k, dt).unwrap();
        a.upload_f32(st, &det(m * k, 7)).unwrap();
        let b = DtypedBuf::zeros(st, k * n, dt).unwrap();
        b.upload_f32(st, &det(k * n, 9)).unwrap();
        let c = DtypedBuf::zeros(st, m * n, dt).unwrap();
        let flop = 2.0 * m as f64 * k as f64 * n as f64;
        let time_us = |rung: &Rung<'_>| -> f64 {
            for _ in 0..10 {
                launch_tile(&ctx, rung, &c, &a, &b, 0, (m, k, n));
            }
            st.synchronize().unwrap();
            let mut best = f64::MAX;
            for _ in 0..4 {
                let s_ev = mk_ev();
                let e_ev = mk_ev();
                s_ev.record(st).unwrap();
                for _ in 0..30 {
                    launch_tile(&ctx, rung, &c, &a, &b, 0, (m, k, n));
                }
                e_ev.record(st).unwrap();
                st.synchronize().unwrap();
                best = best.min(f64::from(s_ev.elapsed_ms(&e_ev).unwrap()) * 1000.0 / 30.0);
            }
            best
        };
        // Alternating order across two rounds.
        let t128a = time_us(&Rung {
            kern: ctx.kernels.gemm_bi_nn_tc128_typed.get(dt),
            tile: (128, 128),
            smem: 71_680,
            threads: 256,
        });
        let twn1 = time_us(&Rung {
            kern: ctx.kernels.gemm_bi_nn_tcwn64_typed.get(dt),
            tile: (128, 256),
            smem: 98_304,
            threads: 256,
        });
        let tw1 = time_us(&Rung {
            kern: ctx.kernels.gemm_bi_nn_tcw64_typed.get(dt),
            tile: (128, 128),
            smem: 65_536,
            threads: 128,
        });
        let twn2 = time_us(&Rung {
            kern: ctx.kernels.gemm_bi_nn_tcwn64_typed.get(dt),
            tile: (128, 256),
            smem: 98_304,
            threads: 256,
        });
        let t128b = time_us(&Rung {
            kern: ctx.kernels.gemm_bi_nn_tc128_typed.get(dt),
            tile: (128, 128),
            smem: 71_680,
            threads: 256,
        });
        let t128 = t128a.min(t128b);
        let tw = tw1;
        let twn = twn1.min(twn2);
        let tf = |t: f64| flop / (t / 1e6) / 1e12;
        println!(
            "M{m:<5} K{k:<5} N{n:<5}: tc128 {t128:7.1}us ({:5.1}TF)  tcw64 {tw:7.1}us  tcwn64 {twn:7.1}us ({:5.1}TF)  wn-ratio {:.3}",
            tf(t128),
            tf(twn),
            t128 / twn
        );
    }
}
