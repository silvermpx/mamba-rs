//! The Mamba-3 MIMO kernels that take another route past a size give the
//! same bits on either route:
//!
//! - the sequential scan with a head's channels in tiles over several
//!   blocks against the whole head in one block, with and without the
//!   replay checkpoints, through the pipelined body and the plain one,
//! - the sequential backward with its rows, keys and tiles in global memory
//!   against the one that holds them in shared memory.
//!
//! Each pair runs on shapes both routes serve, at a compiled rank and a
//! runtime one, in f32, bf16 and f16.
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use cudarc::driver::PushKernelArg;
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer, GpuByteBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_mimo::gpu::kernels::{
    Mamba3MimoKernels, ScanBwdGlobalGeom, ScanGeom, scan_bwd_record_floats, scan_bwd_smem_bytes,
    scan_bwd_threads, scan_smem_bytes,
};

const DTYPES: [WeightDtype; 3] = [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16];

fn det(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s & 0xFF_FFFF) as f32 / 16_777_216.0) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn f32_buf(ctx: &GpuCtx, v: &[f32]) -> GpuBuffer {
    GpuBuffer::from_cpu(&ctx.stream, v).unwrap()
}

fn typed_buf(ctx: &GpuCtx, v: &[f32], dtype: WeightDtype) -> DtypedBuf {
    let b = DtypedBuf::zeros(&ctx.stream, v.len(), dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    b.upload_f32(&ctx.stream, v).unwrap();
    ctx.stream.synchronize().unwrap();
    b
}

fn bits_f32(ctx: &GpuCtx, b: &GpuBuffer) -> Vec<u32> {
    ctx.stream.synchronize().unwrap();
    b.to_cpu(&ctx.stream)
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn bits_typed(ctx: &GpuCtx, b: &DtypedBuf) -> Vec<u32> {
    ctx.stream.synchronize().unwrap();
    let mut v = vec![0f32; b.len_elems()];
    b.download_f32(&ctx.stream, &mut v).unwrap();
    v.iter().map(|x| x.to_bits()).collect()
}

/// A scan shape: sequences, tokens, heads, groups, channels per head,
/// states, rank and rotation angles.
#[derive(Clone, Copy)]
struct Shape {
    batch: usize,
    seq: usize,
    nh: usize,
    ng: usize,
    hd: usize,
    ds: usize,
    rank: usize,
    na: usize,
}

/// A runtime rank over a head of 40 channels, and the compiled rank 4 over
/// two groups and a head of 72.
const SCAN_SHAPES: [Shape; 2] = [
    Shape {
        batch: 2,
        seq: 21,
        nh: 2,
        ng: 1,
        hd: 40,
        ds: 16,
        rank: 3,
        na: 4,
    },
    Shape {
        batch: 2,
        seq: 37,
        nh: 2,
        ng: 2,
        hd: 72,
        ds: 24,
        rank: 4,
        na: 6,
    },
];

/// The whole head in one block (the launch a fitting head always takes)
/// against tiles of 8, 16 and 24 channels, pipelined and plain.
#[test]
fn mimo_scan_tiles_are_the_whole_head() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let k = Mamba3MimoKernels::compile(dev.context(), arch::arch0()).unwrap();
    let s = &ctx.stream;
    let stride = 16usize;
    for Shape {
        batch,
        seq,
        nh,
        ng,
        hd,
        ds,
        rank,
        na,
    } in SCAN_SHAPES
    {
        let rows = batch * seq;
        let di = nh * hd;
        let rgn = rank * ng * ds;
        let ld_proj = 2 * di + 2 * rgn + 3 * nh + na;
        let nseg = seq.div_ceil(stride);
        let bn = f32_buf(&ctx, &det(rows * rgn, 31, 1.0));
        let cn = f32_buf(&ctx, &det(rows * rgn, 32, 1.0));
        let dt_bias = f32_buf(&ctx, &det(nh, 33, 0.5));
        let b_bias = f32_buf(&ctx, &det(nh * rank * ds, 34, 0.5));
        let c_bias = f32_buf(&ctx, &det(nh * rank * ds, 35, 0.5));
        let d = f32_buf(&ctx, &det(nh, 36, 1.0));
        let mx = f32_buf(&ctx, &det(nh * rank * hd, 37, 0.8));
        let mz = f32_buf(&ctx, &det(nh * rank * hd, 38, 0.8));
        let mo = f32_buf(&ctx, &det(nh * rank * hd, 39, 0.8));
        let ssm0 = det(batch * nh * hd * ds, 40, 1.0);
        let k0 = det(batch * nh * rank * ds, 41, 1.0);
        let v0 = det(batch * nh * hd, 42, 1.0);
        let a0: Vec<f32> = det(batch * nh * na, 43, 3.0)
            .iter()
            .map(|x| x + 3.0)
            .collect();
        let whole = ScanGeom {
            tile: hd,
            threads: hd.max(na).next_multiple_of(32).max(256),
            smem: scan_smem_bytes(hd, ds, rank, true),
            pipe: true,
        };
        let tiled = |tile: usize, pipe: bool| ScanGeom {
            tile,
            threads: if pipe {
                (tile.next_multiple_of(32) + na.max(32).next_multiple_of(32)).max(256)
            } else {
                tile.max(na).next_multiple_of(32)
            },
            smem: scan_smem_bytes(tile, ds, rank, pipe),
            pipe,
        };
        let geoms = [
            whole,
            tiled(8, true),
            tiled(16, false),
            tiled(24, true),
            tiled(24, false),
        ];
        for dtype in DTYPES {
            let proj = typed_buf(&ctx, &det(rows * ld_proj, 44, 1.0), dtype);
            for ckpt in [false, true] {
                let run = |g: &ScanGeom| -> Vec<Vec<u32>> {
                    let out = DtypedBuf::zeros(s, rows * di, dtype).unwrap();
                    let (ssm, kst) = (f32_buf(&ctx, &ssm0), f32_buf(&ctx, &k0));
                    let (vst, ang) = (f32_buf(&ctx, &v0), f32_buf(&ctx, &a0));
                    let ck_ssm = GpuBuffer::zeros(s, batch * nseg * nh * hd * ds).unwrap();
                    let ck_k = GpuBuffer::zeros(s, batch * nseg * nh * rank * ds).unwrap();
                    let ck_v = GpuBuffer::zeros(s, batch * nseg * nh * hd).unwrap();
                    let ck_ang = GpuBuffer::zeros(s, batch * nseg * nh * na).unwrap();
                    let done = GpuByteBuffer::zeros(s, batch * nh * 4).unwrap();
                    let ptrs = [
                        out.cached_ptr(),
                        ssm.cached_ptr(),
                        kst.cached_ptr(),
                        vst.cached_ptr(),
                        ang.cached_ptr(),
                        proj.cached_ptr(),
                    ];
                    let ld = ld_proj as i64;
                    let ins = [
                        bn.cached_ptr(),
                        cn.cached_ptr(),
                        dt_bias.cached_ptr(),
                        b_bias.cached_ptr(),
                        c_bias.cached_ptr(),
                        d.cached_ptr(),
                        mx.cached_ptr(),
                        mz.cached_ptr(),
                        mo.cached_ptr(),
                    ];
                    let ints = [
                        seq as i32,
                        nh as i32,
                        ng as i32,
                        hd as i32,
                        ds as i32,
                        rank as i32,
                        na as i32,
                    ];
                    let a_floor = 1e-4f32;
                    let cks = [
                        ck_ssm.cached_ptr(),
                        ck_k.cached_ptr(),
                        ck_v.cached_ptr(),
                        ck_ang.cached_ptr(),
                    ];
                    let (stride_i, tile, pipe) = (stride as i32, g.tile as i32, i32::from(g.pipe));
                    let done_ptr = done.cached_ptr();
                    let kernel = if ckpt { &k.scan_ckpt } else { &k.scan };
                    let mut b = s.launch_builder(kernel.get(dtype));
                    for p in &ptrs {
                        b.arg(p);
                    }
                    b.arg(&ld);
                    for p in &ins {
                        b.arg(p);
                    }
                    for v in &ints {
                        b.arg(v);
                    }
                    b.arg(&a_floor);
                    if ckpt {
                        for p in &cks {
                            b.arg(p);
                        }
                        b.arg(&stride_i);
                    }
                    b.arg(&tile);
                    b.arg(&pipe);
                    b.arg(&done_ptr);
                    unsafe { b.launch(g.cfg(batch, nh, hd)) }.unwrap();
                    let mut got = vec![
                        bits_typed(&ctx, &out),
                        bits_f32(&ctx, &ssm),
                        bits_f32(&ctx, &kst),
                        bits_f32(&ctx, &vst),
                        bits_f32(&ctx, &ang),
                    ];
                    if ckpt {
                        for c in [&ck_ssm, &ck_k, &ck_v, &ck_ang] {
                            got.push(bits_f32(&ctx, c));
                        }
                    }
                    got
                };
                let want = run(&geoms[0]);
                let names = [
                    "rows",
                    "state",
                    "keys",
                    "input",
                    "angles",
                    "ck state",
                    "ck keys",
                    "ck input",
                    "ck angles",
                ];
                for g in &geoms[1..] {
                    let got = run(g);
                    for (i, name) in names.iter().take(want.len()).enumerate() {
                        assert!(
                            got[i] == want[i],
                            "{dtype:?} head {hd} rank {rank} checkpoints {ckpt}, tile {} \
                             pipelined {}: {name}",
                            g.tile,
                            g.pipe
                        );
                    }
                }
            }
        }
    }
}

/// The global sequential backward against mimo_scan_bwd: every group in
/// shared memory, every group in global memory and a mix, a block wider
/// than the head and blocks the head is wider than (whose block sums run
/// their upper levels on each thread's own leaves). The replay records are
/// compared too.
#[test]
fn mimo_scan_bwd_global_is_the_shared_kernel() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let k = Mamba3MimoKernels::compile(dev.context(), arch::arch0()).unwrap();
    let s = &ctx.stream;
    let stride = 16usize;
    for Shape {
        batch,
        seq,
        nh,
        ng,
        hd,
        ds,
        rank,
        na,
    } in SCAN_SHAPES
    {
        let rows = batch * seq;
        let di = nh * hd;
        let rgn = rank * ng * ds;
        let rn = rank * ds;
        let ld_proj = 2 * di + 2 * rgn + 3 * nh + na;
        let nseg = seq.div_ceil(stride);
        let lanes = hd.next_multiple_of(32);
        let bn = f32_buf(&ctx, &det(rows * rgn, 51, 1.0));
        let cn = f32_buf(&ctx, &det(rows * rgn, 52, 1.0));
        let ck_ssm = f32_buf(&ctx, &det(batch * nseg * nh * hd * ds, 53, 1.0));
        let ck_k = f32_buf(&ctx, &det(batch * nseg * nh * rn, 54, 1.0));
        let ck_v = f32_buf(&ctx, &det(batch * nseg * nh * hd, 55, 1.0));
        let ck_ang: Vec<f32> = det(batch * nseg * nh * na, 56, 3.0)
            .iter()
            .map(|x| x + 3.0)
            .collect();
        let ck_ang = f32_buf(&ctx, &ck_ang);
        let dt_bias = f32_buf(&ctx, &det(nh, 57, 0.5));
        let b_bias = f32_buf(&ctx, &det(nh * rn, 58, 0.5));
        let c_bias = f32_buf(&ctx, &det(nh * rn, 59, 0.5));
        let d = f32_buf(&ctx, &det(nh, 60, 1.0));
        let mx = f32_buf(&ctx, &det(nh * rank * hd, 61, 0.8));
        let mz = f32_buf(&ctx, &det(nh * rank * hd, 62, 0.8));
        let mo = f32_buf(&ctx, &det(nh * rank * hd, 63, 0.8));
        let geom = |threads: usize, (ds_sh, kg_sh, pr_sh): (bool, bool, bool)| {
            let wide = lanes > threads;
            let leaves = if wide { threads } else { lanes };
            let mut floats = 3 * na + 3 * leaves;
            if ds_sh {
                floats += (hd * (ds + 1)).next_multiple_of(4);
            }
            if kg_sh {
                floats += 5 * rn;
            }
            if pr_sh {
                floats += 3 * hd * rank;
            }
            ScanBwdGlobalGeom {
                threads,
                ds_shared: ds_sh,
                keys_shared: kg_sh,
                tiles_shared: pr_sh,
                wide,
                smem: floats * 4,
            }
        };
        let geoms = [
            geom(256, (true, true, true)),
            geom(256, (false, false, false)),
            geom(256, (true, false, true)),
            geom(64, (false, true, false)),
            geom(32, (true, true, true)),
            geom(32, (false, false, false)),
        ];
        for dtype in DTYPES {
            let proj = typed_buf(&ctx, &det(rows * ld_proj, 64, 1.0), dtype);
            let d_o = typed_buf(&ctx, &det(rows * di, 65, 1.0), dtype);
            // None runs mimo_scan_bwd, Some(geometry) the global kernel.
            let run = |g: Option<&ScanBwdGlobalGeom>| -> Vec<Vec<u32>> {
                let z = |n: usize| GpuBuffer::zeros(s, n.max(1)).unwrap();
                let d_proj = DtypedBuf::zeros(s, rows * ld_proj, dtype).unwrap();
                let (d_bn_h, d_cn_h) = (z(rows * nh * rn), z(rows * nh * rn));
                let d_tanh_h = z(rows * nh * na);
                let (g_x, g_z, g_o) = (
                    z(batch * nh * rank * hd),
                    z(batch * nh * rank * hd),
                    z(batch * nh * rank * hd),
                );
                let (g_d, g_dtb) = (z(batch * nh), z(batch * nh));
                let (g_bb, g_cb) = (z(batch * nh * rn), z(batch * nh * rn));
                let rec = z(batch * nh * stride * scan_bwd_record_floats(hd, ds, rank, na));
                let per_bh = batch * nh;
                let scratch = g.map(|g| {
                    let buf = |global: bool, n: usize| global.then(|| z(n));
                    [
                        buf(!g.ds_shared, per_bh * hd * ds),
                        buf(!g.keys_shared, per_bh * 5 * rn),
                        buf(!g.tiles_shared, per_bh * 3 * hd * rank),
                        buf(true, per_bh * 2 * hd),
                        buf(g.wide, per_bh * 4 * lanes),
                    ]
                });
                let outs = [
                    d_bn_h.cached_ptr(),
                    d_cn_h.cached_ptr(),
                    d_tanh_h.cached_ptr(),
                    g_x.cached_ptr(),
                    g_z.cached_ptr(),
                    g_o.cached_ptr(),
                    g_d.cached_ptr(),
                    g_dtb.cached_ptr(),
                    g_bb.cached_ptr(),
                    g_cb.cached_ptr(),
                    rec.cached_ptr(),
                ];
                let ins = [
                    d_o.cached_ptr(),
                    proj.cached_ptr(),
                    bn.cached_ptr(),
                    cn.cached_ptr(),
                    ck_ssm.cached_ptr(),
                    ck_k.cached_ptr(),
                    ck_v.cached_ptr(),
                    ck_ang.cached_ptr(),
                    dt_bias.cached_ptr(),
                    b_bias.cached_ptr(),
                    c_bias.cached_ptr(),
                    d.cached_ptr(),
                    mx.cached_ptr(),
                    mz.cached_ptr(),
                    mo.cached_ptr(),
                ];
                let scratch_ptrs: Option<Vec<u64>> = scratch.as_ref().map(|bufs| {
                    bufs.iter()
                        .map(|b| b.as_ref().map_or(0u64, |b| b.cached_ptr()))
                        .collect()
                });
                let ld = ld_proj as i64;
                let ints = [
                    seq as i32,
                    stride as i32,
                    nh as i32,
                    ng as i32,
                    hd as i32,
                    ds as i32,
                    rank as i32,
                    na as i32,
                ];
                let a_floor = 1e-4f32;
                let d_proj_ptr = d_proj.cached_ptr();
                let kernel = if g.is_some() {
                    &k.scan_bwd_global
                } else {
                    &k.scan_bwd
                };
                let mut b = s.launch_builder(kernel.get(dtype));
                b.arg(&d_proj_ptr);
                b.arg(&ld);
                for p in &outs {
                    b.arg(p);
                }
                if let Some(ptrs) = &scratch_ptrs {
                    for p in ptrs {
                        b.arg(p);
                    }
                }
                for p in &ins {
                    b.arg(p);
                }
                for v in &ints {
                    b.arg(v);
                }
                b.arg(&a_floor);
                let (threads, smem) = match g {
                    None => (
                        scan_bwd_threads(hd),
                        scan_bwd_smem_bytes((hd, ds, rank, na)),
                    ),
                    Some(g) => (g.threads, g.smem),
                };
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (batch as u32, nh as u32, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: smem as u32,
                };
                unsafe { b.launch(cfg) }.unwrap();
                vec![
                    bits_typed(&ctx, &d_proj),
                    bits_f32(&ctx, &d_bn_h),
                    bits_f32(&ctx, &d_cn_h),
                    bits_f32(&ctx, &d_tanh_h),
                    bits_f32(&ctx, &g_x),
                    bits_f32(&ctx, &g_z),
                    bits_f32(&ctx, &g_o),
                    bits_f32(&ctx, &g_d),
                    bits_f32(&ctx, &g_dtb),
                    bits_f32(&ctx, &g_bb),
                    bits_f32(&ctx, &g_cb),
                    bits_f32(&ctx, &rec),
                ]
            };
            let want = run(None);
            let names = [
                "d_proj",
                "dB",
                "dC",
                "d tanh",
                "g x",
                "g z",
                "g o",
                "g D",
                "g dt bias",
                "g b bias",
                "g c bias",
                "record",
            ];
            for g in &geoms {
                let got = run(Some(g));
                for (i, name) in names.iter().enumerate() {
                    assert!(
                        got[i] == want[i],
                        "{dtype:?} head {hd} rank {rank}, block {} groups shared {} {} {}: {name}",
                        g.threads,
                        g.ds_shared,
                        g.keys_shared,
                        g.tiles_shared
                    );
                }
            }
        }
    }
}
