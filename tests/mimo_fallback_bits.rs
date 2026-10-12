//! The Mamba-3 MIMO kernels that take another route past a size give the
//! same bits on either route: the sequential scan with a head's channels in
//! tiles over several blocks against the whole head in one block, with and
//! without the replay checkpoints, through the pipelined body and the plain
//! one. Each pair runs on shapes both routes serve, at a compiled rank and
//! a runtime one, in f32, bf16 and f16.
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use cudarc::driver::PushKernelArg;
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer, GpuByteBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_mimo::gpu::kernels::{Mamba3MimoKernels, ScanGeom, scan_smem_bytes};

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
