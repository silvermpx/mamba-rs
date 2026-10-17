//! The Mamba-3 MIMO kernels that take another route past a size give the
//! same bits on either route:
//!
//! - the sequential scan with a head's channels in tiles over several
//!   blocks against the whole head in one block, with and without the
//!   replay checkpoints, through the pipelined body and the plain one,
//! - the sequential backward with its rows, keys and tiles in global memory
//!   against the one that holds them in shared memory,
//! - the chunked path's chunk state, chunk scan and local state gradient
//!   with the head's channels in tiles against the whole head, the state
//!   terms and pair terms staging their rows in global memory against
//!   shared memory, and the entering lead over a block narrower than the
//!   head against a thread per channel,
//! - the register-tile chunk state and local state gradient, the whole head
//!   in a block and in channel groups, against the channel-parallel kernels,
//! - the chunked backward's token-tile inter and state terms against the
//!   per-token kernels, at heads and state widths past one pass of 128
//!   columns as well as within one,
//! - the chunked backward's angle walk, a batch of tokens staged at a time,
//!   against the walk as it was released, a token at a time (frozen in
//!   `fixtures/m3m_angles_reference.cu`).
//!
//! Each pair runs on shapes both routes serve, at a compiled rank and a
//! runtime one, in f32, bf16 and f16.
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;
#[path = "common/frozen_kernels.rs"]
mod frozen_kernels;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use frozen_kernels::frozen_module;
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer, GpuByteBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{y_runs, yz_runs};
use mamba_rs::mamba3_mimo::gpu::kernels::{
    Mamba3MimoKernels, ScanBwdGlobalGeom, ScanGeom, angle_walk_smem_bytes, chunk_scan_tiles_cfg,
    chunked_geom, scan_bwd_record_floats, scan_bwd_smem_bytes, scan_bwd_threads, scan_smem_bytes,
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
        let k = Mamba3MimoKernels::compile(dev.context(), arch::arch0(), rank).unwrap();
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
                    let y = GpuBuffer::zeros(s, rows * nh * rank * hd).unwrap();
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
                    let (done_ptr, y_ptr) = (done.cached_ptr(), y.cached_ptr());
                    let kernel = if ckpt { &k.scan_ckpt } else { &k.scan };
                    for (y_first, z_first, cfg) in yz_runs(g.cfg(batch, nh, hd)) {
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
                        b.arg(&y_ptr);
                        b.arg(&y_first);
                        b.arg(&z_first);
                        b.arg(&0u64);
                        unsafe { b.launch(cfg) }.unwrap();
                    }
                    let mut got = vec![
                        bits_typed(&ctx, &out),
                        bits_f32(&ctx, &y),
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
                    "readouts",
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
        let k = Mamba3MimoKernels::compile(dev.context(), arch::arch0(), rank).unwrap();
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
            let dy_given = f32_buf(&ctx, &det(rows * nh * rank * hd, 66, 1.0));
            // None runs mimo_scan_bwd, Some(geometry) the global kernel. A
            // given readout gradient skips the gate, as the output norm does.
            let run = |g: Option<&ScanBwdGlobalGeom>, dy_in: u64| -> Vec<Vec<u32>> {
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
                for (y_first, cfg) in y_runs(cfg) {
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
                    b.arg(&dy_in);
                    b.arg(&y_first);
                    b.arg(&0u64);
                    unsafe { b.launch(cfg) }.unwrap();
                }
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
            for dy_in in [0u64, dy_given.cached_ptr()] {
                let want = run(None, dy_in);
                for g in &geoms {
                    let got = run(Some(g), dy_in);
                    for (i, name) in names.iter().enumerate() {
                        assert!(
                            got[i] == want[i],
                            "{dtype:?} head {hd} rank {rank}, given dy {}, block {} groups \
                             shared {} {} {}: {name}",
                            dy_in != 0,
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
}

/// A kernel argument of any of the kinds the chunked kernels take.
#[derive(Clone, Copy)]
enum Arg {
    P(u64),
    L(i64),
    I(i32),
}

fn launch(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    let mut b = ctx.stream.launch_builder(f);
    for a in args {
        match a {
            Arg::P(v) => {
                b.arg(v);
            }
            Arg::L(v) => {
                b.arg(v);
            }
            Arg::I(v) => {
                b.arg(v);
            }
        }
    }
    unsafe { b.launch(cfg) }.unwrap();
}

/// No packing (see `mamba_rs::varlen`): the windows here hold one sequence
/// per row.
const NO_PACKING: Arg = Arg::P(0);

/// [`launch`] once per run of the grid's y axis, each run's first block
/// appended, then `tail` (the kernel's packing and what follows it).
fn launch_runs(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg], tail: &[Arg]) {
    for (y_first, cfg) in y_runs(cfg) {
        let mut run = args.to_vec();
        run.push(Arg::I(y_first));
        run.extend_from_slice(tail);
        launch(ctx, f, cfg, &run);
    }
}

/// [`launch`] once per run of the grid's y and z axes, each run's first y
/// and z blocks appended, then no packing.
fn launch_yz_runs(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    for (y_first, z_first, cfg) in yz_runs(cfg) {
        let mut run = args.to_vec();
        run.extend([Arg::I(y_first), Arg::I(z_first), NO_PACKING]);
        launch(ctx, f, cfg, &run);
    }
}

fn cfg(grid: (usize, usize, usize), block: usize, smem: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.0 as u32, grid.1 as u32, grid.2 as u32),
        block_dim: (block as u32, 1, 1),
        shared_mem_bytes: smem as u32,
    }
}

/// The inputs every chunked kernel here reads, for one shape, drawn once:
/// two sequences of 100 tokens, a whole chunk and a partial one.
struct ChunkedInputs {
    seq: usize,
    nc: usize,
    ld_proj: usize,
    bn: GpuBuffer,
    cn: GpuBuffer,
    coef: GpuBuffer,
    sn: GpuBuffer,
    cs: GpuBuffer,
    b_bias: GpuBuffer,
    c_bias: GpuBuffer,
    d: GpuBuffer,
    mx: GpuBuffer,
    mz: GpuBuffer,
    mo: GpuBuffer,
    enter: GpuBuffer,
    dy: GpuBuffer,
    dst: GpuBuffer,
}

impl ChunkedInputs {
    fn new(ctx: &GpuCtx, sh: Shape) -> Self {
        let seq = 100usize;
        let rows = sh.batch * seq;
        let nc = seq.div_ceil(64);
        let di = sh.nh * sh.hd;
        let rgn = sh.rank * sh.ng * sh.ds;
        let per = sh.batch * nc * sh.nh * sh.hd * sh.ds;
        Self {
            seq,
            nc,
            ld_proj: 2 * di + 2 * rgn + 3 * sh.nh + sh.na,
            bn: f32_buf(ctx, &det(rows * rgn, 71, 1.0)),
            cn: f32_buf(ctx, &det(rows * rgn, 72, 1.0)),
            coef: f32_buf(ctx, &det(rows * sh.nh * 4, 73, 1.0)),
            sn: f32_buf(ctx, &det(rows * sh.nh * sh.na, 74, 1.0)),
            cs: f32_buf(ctx, &det(rows * sh.nh * sh.na, 75, 1.0)),
            b_bias: f32_buf(ctx, &det(sh.nh * sh.rank * sh.ds, 76, 0.5)),
            c_bias: f32_buf(ctx, &det(sh.nh * sh.rank * sh.ds, 77, 0.5)),
            d: f32_buf(ctx, &det(sh.nh, 78, 1.0)),
            mx: f32_buf(ctx, &det(sh.nh * sh.rank * sh.hd, 79, 0.8)),
            mz: f32_buf(ctx, &det(sh.nh * sh.rank * sh.hd, 80, 0.8)),
            mo: f32_buf(ctx, &det(sh.nh * sh.rank * sh.hd, 81, 0.8)),
            enter: f32_buf(ctx, &det(per, 82, 1.0)),
            dy: f32_buf(ctx, &det(rows * sh.nh * sh.rank * sh.hd, 83, 1.0)),
            dst: f32_buf(ctx, &det(per, 84, 1.0)),
        }
    }
}

/// The chunk state, the chunk scan and the local state gradient with the
/// head in channel tiles of 8 and 16 against the whole head.
#[test]
fn mimo_chunked_tiles_are_the_whole_head() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let s = &ctx.stream;
    for sh in SCAN_SHAPES {
        let k = Mamba3MimoKernels::compile(dev.context(), arch::arch0(), sh.rank).unwrap();
        let inp = ChunkedInputs::new(&ctx, sh);
        let (b, nh, hd, ds, rank) = (sh.batch, sh.nh, sh.hd, sh.ds, sh.rank);
        let rows = b * inp.seq;
        let ints = |extra: &[i32]| -> Vec<Arg> {
            [
                inp.seq as i32,
                nh as i32,
                sh.ng as i32,
                hd as i32,
                ds as i32,
                rank as i32,
                sh.na as i32,
            ]
            .iter()
            .chain(extra)
            .map(|&v| Arg::I(v))
            .collect()
        };
        for dtype in DTYPES {
            let proj = typed_buf(&ctx, &det(rows * inp.ld_proj, 85, 1.0), dtype);
            let mut runs = Vec::new();
            for pt in [hd, 8, 16] {
                let tiles = hd.div_ceil(pt);
                // The chunk state, one state index and 16 tokens a pass.
                let contrib = GpuBuffer::zeros(s, b * inp.nc * nh * hd * ds).unwrap();
                let (nt, sl) = (1usize, 16usize);
                let mut args = vec![
                    Arg::P(contrib.cached_ptr()),
                    Arg::P(proj.cached_ptr()),
                    Arg::L(inp.ld_proj as i64),
                    Arg::P(inp.bn.cached_ptr()),
                    Arg::P(inp.coef.cached_ptr()),
                    Arg::P(inp.sn.cached_ptr()),
                    Arg::P(inp.cs.cached_ptr()),
                    Arg::P(inp.b_bias.cached_ptr()),
                    Arg::P(inp.mx.cached_ptr()),
                ];
                args.extend(ints(&[nt as i32, sl as i32, pt as i32]));
                let smem = (sl * rank * nt + sl * pt + sl) * 4;
                launch_yz_runs(
                    &ctx,
                    k.chunk_state.get(dtype),
                    cfg((b * inp.nc, nh, ds.div_ceil(nt) * tiles), nt * pt, smem),
                    &args,
                );
                // The chunk scan, two query tokens and four key tokens a pass.
                let out = DtypedBuf::zeros(s, rows * nh * hd, dtype).unwrap();
                let y_out = GpuBuffer::zeros(s, rows * nh * rank * hd).unwrap();
                let (tt, st) = (2usize, 4usize);
                let mut args = vec![
                    Arg::P(out.cached_ptr()),
                    Arg::P(y_out.cached_ptr()),
                    Arg::P(proj.cached_ptr()),
                    Arg::L(inp.ld_proj as i64),
                    Arg::P(inp.bn.cached_ptr()),
                    Arg::P(inp.cn.cached_ptr()),
                    Arg::P(inp.coef.cached_ptr()),
                    Arg::P(inp.sn.cached_ptr()),
                    Arg::P(inp.cs.cached_ptr()),
                    Arg::P(inp.enter.cached_ptr()),
                    Arg::P(inp.b_bias.cached_ptr()),
                    Arg::P(inp.c_bias.cached_ptr()),
                    Arg::P(inp.d.cached_ptr()),
                    Arg::P(inp.mx.cached_ptr()),
                    Arg::P(inp.mz.cached_ptr()),
                    Arg::P(inp.mo.cached_ptr()),
                ];
                args.extend(ints(&[tt as i32, st as i32, pt as i32]));
                let smem =
                    (tt * rank * ds + st * rank * (ds + 1) + tt * rank * st * rank + st * pt) * 4;
                launch_yz_runs(
                    &ctx,
                    k.chunk_scan.get(dtype),
                    cfg(
                        (b * inp.nc, nh, 64usize.div_ceil(tt) * tiles),
                        (tt * pt).next_multiple_of(32),
                        smem,
                    ),
                    &args,
                );
                // The local state gradient, two state indices a block.
                let dsl = GpuBuffer::zeros(s, b * inp.nc * nh * hd * ds).unwrap();
                let nt = 2usize;
                let mut args = vec![
                    Arg::P(dsl.cached_ptr()),
                    Arg::P(inp.dy.cached_ptr()),
                    Arg::P(inp.cn.cached_ptr()),
                    Arg::P(inp.c_bias.cached_ptr()),
                    Arg::P(inp.coef.cached_ptr()),
                    Arg::P(inp.sn.cached_ptr()),
                    Arg::P(inp.cs.cached_ptr()),
                ];
                args.extend(ints(&[nt as i32, pt as i32]));
                launch_yz_runs(
                    &ctx,
                    &k.cb.dsloc,
                    cfg(
                        (b * inp.nc, nh, ds.div_ceil(nt) * tiles),
                        nt * pt,
                        (64 * rank * nt + 64) * 4,
                    ),
                    &args,
                );
                runs.push([
                    bits_f32(&ctx, &contrib),
                    bits_typed(&ctx, &out),
                    bits_f32(&ctx, &y_out),
                    bits_f32(&ctx, &dsl),
                ]);
            }
            for (i, name) in ["chunk state", "chunk scan rows", "readouts", "local dS"]
                .iter()
                .enumerate()
            {
                for (run, pt) in runs.iter().zip([hd, 8, 16]).skip(1) {
                    assert!(
                        run[i] == runs[0][i],
                        "{dtype:?} head {hd} rank {rank}, channel tile {pt}: {name}"
                    );
                }
            }
        }
    }
}

/// The register-tile chunk state and local state gradient against the
/// channel-parallel kernels, with the head in one block where the tile
/// kernel's registers allow it and in groups of 4, 8 and 12 channels: the
/// groups a block narrower than the head takes on a GPU whose registers do
/// not hold it.
#[test]
fn mimo_chunked_register_tiles_are_the_channel_parallel_kernels() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let s = &ctx.stream;
    for sh in SCAN_SHAPES {
        let k = Mamba3MimoKernels::compile(dev.context(), arch::arch0(), sh.rank).unwrap();
        let inp = ChunkedInputs::new(&ctx, sh);
        let (b, nh, hd, ds, rank) = (sh.batch, sh.nh, sh.hd, sh.ds, sh.rank);
        let rows = b * inp.seq;
        let ints = |extra: &[i32]| -> Vec<Arg> {
            [
                inp.seq as i32,
                nh as i32,
                sh.ng as i32,
                hd as i32,
                ds as i32,
                rank as i32,
                sh.na as i32,
            ]
            .iter()
            .chain(extra)
            .map(|&v| Arg::I(v))
            .collect()
        };
        let whole = hd.next_multiple_of(4);
        let mut groups = vec![4usize, 8, 12];
        if 8 * whole <= k.chunk_threads.state_tiles.min(k.chunk_threads.dsloc_tiles) {
            groups.push(whole);
        }
        let jc = 1usize.max(ds.div_ceil(32).next_power_of_two().min(4));
        let sl = 16usize;
        for dtype in DTYPES {
            let proj = typed_buf(&ctx, &det(rows * inp.ld_proj, 86, 1.0), dtype);
            let state_args = |out: &GpuBuffer| {
                vec![
                    Arg::P(out.cached_ptr()),
                    Arg::P(proj.cached_ptr()),
                    Arg::L(inp.ld_proj as i64),
                    Arg::P(inp.bn.cached_ptr()),
                    Arg::P(inp.coef.cached_ptr()),
                    Arg::P(inp.sn.cached_ptr()),
                    Arg::P(inp.cs.cached_ptr()),
                    Arg::P(inp.b_bias.cached_ptr()),
                    Arg::P(inp.mx.cached_ptr()),
                ]
            };
            let dsloc_args = |out: &GpuBuffer| {
                vec![
                    Arg::P(out.cached_ptr()),
                    Arg::P(inp.dy.cached_ptr()),
                    Arg::P(inp.cn.cached_ptr()),
                    Arg::P(inp.c_bias.cached_ptr()),
                    Arg::P(inp.coef.cached_ptr()),
                    Arg::P(inp.sn.cached_ptr()),
                    Arg::P(inp.cs.cached_ptr()),
                ]
            };
            let per = b * inp.nc * nh * hd * ds;
            // The channel-parallel kernels over the whole head.
            let contrib = GpuBuffer::zeros(s, per).unwrap();
            let mut args = state_args(&contrib);
            args.extend(ints(&[1, sl as i32, hd as i32]));
            launch_yz_runs(
                &ctx,
                k.chunk_state.get(dtype),
                cfg((b * inp.nc, nh, ds), hd, (sl * rank + sl * hd + sl) * 4),
                &args,
            );
            let dsl = GpuBuffer::zeros(s, per).unwrap();
            let mut args = dsloc_args(&dsl);
            args.extend(ints(&[1, hd as i32]));
            launch_yz_runs(
                &ctx,
                &k.cb.dsloc,
                cfg((b * inp.nc, nh, ds), hd, (64 * rank + 64) * 4),
                &args,
            );
            let want = [bits_f32(&ctx, &contrib), bits_f32(&ctx, &dsl)];
            for &pg in &groups {
                let grid = (b * inp.nc, nh, ds.div_ceil(32 * jc) * hd.div_ceil(pg));
                let smem = (sl * rank * (pg + 32 * jc) + sl) * 4;
                let tile_ints = [jc as i32, sl as i32, pg as i32];
                let contrib = GpuBuffer::zeros(s, per).unwrap();
                let mut args = state_args(&contrib);
                args.extend(ints(&tile_ints));
                launch_yz_runs(
                    &ctx,
                    k.chunk_state_tiles.get(dtype),
                    cfg(grid, 8 * pg, smem),
                    &args,
                );
                let dsl = GpuBuffer::zeros(s, per).unwrap();
                let mut args = dsloc_args(&dsl);
                args.extend(ints(&tile_ints));
                launch_yz_runs(&ctx, &k.cb.dsloc_tiles, cfg(grid, 8 * pg, smem), &args);
                for (i, (name, got)) in [("chunk state", &contrib), ("local dS", &dsl)]
                    .iter()
                    .enumerate()
                {
                    assert!(
                        bits_f32(&ctx, got) == want[i],
                        "{dtype:?} head {hd} rank {rank}, groups of {pg}: {name}"
                    );
                }
            }
        }
    }
}

/// The state terms and the pair terms with their rows staged in a global
/// scratch against shared memory, and the entering lead over a block of 32
/// threads (the tree's upper levels on each thread's own leaves) against a
/// thread per channel.
#[test]
fn mimo_chunked_staging_in_global_is_the_shared_staging() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let s = &ctx.stream;
    for sh in SCAN_SHAPES {
        let k = Mamba3MimoKernels::compile(dev.context(), arch::arch0(), sh.rank).unwrap();
        let inp = ChunkedInputs::new(&ctx, sh);
        let (b, nh, hd, ds, rank, na) = (sh.batch, sh.nh, sh.hd, sh.ds, sh.rank, sh.na);
        let rows = b * inp.seq;
        let rn = rank * ds;
        let ints = |extra: &[i32]| -> Vec<Arg> {
            [
                inp.seq as i32,
                nh as i32,
                sh.ng as i32,
                hd as i32,
                ds as i32,
                rank as i32,
                na as i32,
            ]
            .iter()
            .chain(extra)
            .map(|&v| Arg::I(v))
            .collect()
        };
        let dk0 = det(rows * nh * rn, 86, 1.0);
        let du0 = det(rows * nh * rank * hd, 87, 1.0);
        let dq0 = det(rows * nh * rn, 88, 1.0);
        for dtype in DTYPES {
            let proj = typed_buf(&ctx, &det(rows * inp.ld_proj, 89, 1.0), dtype);
            // The state terms: eight tokens a block, dS staged or not.
            let tg = 8usize;
            let blocks = b * inp.nc * nh * 64usize.div_ceil(tg);
            let mut terms = Vec::new();
            for (stage, glob) in [(1, false), (1, true), (0, false), (0, true)] {
                let (dk, du) = (f32_buf(&ctx, &dk0), f32_buf(&ctx, &du0));
                let dcs = GpuBuffer::zeros(s, rows * nh).unwrap();
                let scratch = glob.then(|| GpuBuffer::zeros(s, blocks * (rn + rank * hd)).unwrap());
                let base = if glob { 0 } else { (rn + rank * hd) * 4 };
                let smem = base + if stage == 1 { hd * (ds + 1) * 4 } else { 0 };
                let mut args = vec![
                    Arg::P(dk.cached_ptr()),
                    Arg::P(du.cached_ptr()),
                    Arg::P(dcs.cached_ptr()),
                    Arg::P(inp.dst.cached_ptr()),
                    Arg::P(proj.cached_ptr()),
                    Arg::L(inp.ld_proj as i64),
                    Arg::P(inp.bn.cached_ptr()),
                    Arg::P(inp.b_bias.cached_ptr()),
                    Arg::P(inp.coef.cached_ptr()),
                    Arg::P(inp.sn.cached_ptr()),
                    Arg::P(inp.cs.cached_ptr()),
                    Arg::P(inp.mx.cached_ptr()),
                    Arg::P(scratch.as_ref().map_or(0, |g| g.cached_ptr())),
                ];
                args.extend(ints(&[tg as i32, stage]));
                launch_runs(
                    &ctx,
                    k.cb.state_terms.get(dtype),
                    cfg((b * inp.nc, nh, 64usize.div_ceil(tg)), 256, smem),
                    &args,
                    &[NO_PACKING],
                );
                terms.push([
                    bits_f32(&ctx, &dk),
                    bits_f32(&ctx, &du),
                    bits_f32(&ctx, &dcs),
                ]);
            }
            for (i, t) in terms.iter().enumerate().skip(1) {
                assert!(*t == terms[0], "{dtype:?} head {hd}: state terms, run {i}");
            }
            // The pair terms at a tile of two tokens.
            let tl = 2usize;
            let tr = tl * rank;
            let mut pairs = Vec::new();
            for glob in [false, true] {
                let (dq, dk, du) = (
                    f32_buf(&ctx, &dq0),
                    f32_buf(&ctx, &dk0),
                    f32_buf(&ctx, &du0),
                );
                let dwp = GpuBuffer::zeros(s, b * inp.nc * nh * 64 * 65 / 2).unwrap();
                let scratch = glob.then(|| {
                    GpuBuffer::zeros(s, b * inp.nc * nh * tr * (3 * ds + 2 * hd + 2)).unwrap()
                });
                let smem = if glob {
                    3 * tr * tr * 4
                } else {
                    (3 * tr * ds + 2 * tr * hd + 2 * tr + 3 * tr * tr) * 4
                };
                let mut args = vec![
                    Arg::P(dq.cached_ptr()),
                    Arg::P(dk.cached_ptr()),
                    Arg::P(du.cached_ptr()),
                    Arg::P(dwp.cached_ptr()),
                    Arg::P(inp.dy.cached_ptr()),
                    Arg::P(proj.cached_ptr()),
                    Arg::L(inp.ld_proj as i64),
                    Arg::P(inp.bn.cached_ptr()),
                    Arg::P(inp.cn.cached_ptr()),
                    Arg::P(inp.b_bias.cached_ptr()),
                    Arg::P(inp.c_bias.cached_ptr()),
                    Arg::P(inp.coef.cached_ptr()),
                    Arg::P(inp.sn.cached_ptr()),
                    Arg::P(inp.cs.cached_ptr()),
                    Arg::P(inp.mx.cached_ptr()),
                    Arg::P(scratch.as_ref().map_or(0, |g| g.cached_ptr())),
                ];
                args.extend(ints(&[tl as i32]));
                launch_runs(
                    &ctx,
                    k.cb.pairs.get(dtype),
                    cfg((b * inp.nc, nh, 1), 256, smem),
                    &args,
                    &[NO_PACKING],
                );
                pairs.push([
                    bits_f32(&ctx, &dq),
                    bits_f32(&ctx, &dk),
                    bits_f32(&ctx, &du),
                    bits_f32(&ctx, &dwp),
                ]);
            }
            assert!(
                pairs[1] == pairs[0],
                "{dtype:?} head {hd} rank {rank}: pair terms"
            );
        }
        // The entering lead, a thread per channel against a block of 32.
        let lanes = hd.next_multiple_of(32);
        let d_init = f32_buf(&ctx, &det(b * nh * hd * ds, 90, 1.0));
        let e0 = f32_buf(&ctx, &det(b * nh, 91, 1.0));
        let kst = f32_buf(&ctx, &det(b * nh * rn, 92, 1.0));
        let vst = f32_buf(&ctx, &det(b * nh * hd, 93, 1.0));
        let mut leads = Vec::new();
        for threads in [lanes, 32] {
            let de0 = GpuBuffer::zeros(s, b * nh).unwrap();
            let gpsi = GpuBuffer::zeros(s, b * nh * rank * hd).unwrap();
            let leaves = (lanes > threads).then(|| GpuBuffer::zeros(s, b * nh * lanes).unwrap());
            let args = [
                Arg::P(de0.cached_ptr()),
                Arg::P(gpsi.cached_ptr()),
                Arg::P(d_init.cached_ptr()),
                Arg::P(e0.cached_ptr()),
                Arg::P(kst.cached_ptr()),
                Arg::P(vst.cached_ptr()),
                Arg::P(inp.mx.cached_ptr()),
                Arg::P(leaves.as_ref().map_or(0, |g| g.cached_ptr())),
                Arg::I(nh as i32),
                Arg::I(hd as i32),
                Arg::I(ds as i32),
                Arg::I(rank as i32),
            ];
            launch_runs(
                &ctx,
                &k.cb.lead,
                cfg((b, nh, 1), threads, lanes.min(threads) * 4),
                &args,
                &[NO_PACKING, Arg::I(inp.seq as i32)],
            );
            leads.push([bits_f32(&ctx, &de0), bits_f32(&ctx, &gpsi)]);
        }
        assert!(leads[1] == leads[0], "head {hd}: the entering lead");
    }
}

/// Shapes past one pass of 128 columns: heads of 160 and 136 channels and
/// states of 160 and 136 indices, beside the narrow ones.
const WIDE_SHAPES: [Shape; 3] = [
    Shape {
        batch: 1,
        seq: 100,
        nh: 2,
        ng: 1,
        hd: 160,
        ds: 48,
        rank: 2,
        na: 12,
    },
    Shape {
        batch: 1,
        seq: 100,
        nh: 2,
        ng: 2,
        hd: 48,
        ds: 160,
        rank: 2,
        na: 40,
    },
    Shape {
        batch: 1,
        seq: 100,
        nh: 1,
        ng: 1,
        hd: 136,
        ds: 136,
        rank: 1,
        na: 34,
    },
];

/// The forward scan's register tiles against the channel-parallel scan, and
/// the token-tile inter term and state terms against the per-token kernels
/// (the state terms with every row in global memory): the same bits at the
/// tiles the trainer plans, at heads and state widths within one pass of
/// 128 columns and past it (a head past 128 channels in groups of them).
#[test]
fn mimo_chunked_token_tiles_are_the_per_token_kernels() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let s = &ctx.stream;
    let mut checked = 0;
    for sh in SCAN_SHAPES.into_iter().chain(WIDE_SHAPES) {
        let k = Mamba3MimoKernels::compile(dev.context(), arch::arch0(), sh.rank).unwrap();
        let inp = ChunkedInputs::new(&ctx, sh);
        let (b, nh, hd, ds, rank, na) = (sh.batch, sh.nh, sh.hd, sh.ds, sh.rank, sh.na);
        let geom = chunked_geom((hd, ds, rank), k.smem_optin, true, k.chunk_threads).unwrap();
        let ((tt_i, inter_smem), (tt_t, terms_smem)) = (geom.inter_tiles, geom.terms_tiles);
        assert!(
            tt_i > 0 && tt_t > 0,
            "head {hd} state {ds} rank {rank}: no token tile planned: {geom:?}"
        );
        let rows = b * inp.seq;
        let rn = rank * ds;
        let ints = |extra: &[i32]| -> Vec<Arg> {
            [
                inp.seq as i32,
                nh as i32,
                sh.ng as i32,
                hd as i32,
                ds as i32,
                rank as i32,
                na as i32,
            ]
            .iter()
            .chain(extra)
            .map(|&v| Arg::I(v))
            .collect()
        };
        let label = format!("head {hd} state {ds} rank {rank}");
        // The inter-chunk term: dq and I per token.
        let inter_run = |tiles: bool| {
            let dq = GpuBuffer::zeros(s, rows * nh * rn).unwrap();
            let inter = GpuBuffer::zeros(s, rows * nh).unwrap();
            let mut args = vec![
                Arg::P(dq.cached_ptr()),
                Arg::P(inter.cached_ptr()),
                Arg::P(inp.dy.cached_ptr()),
                Arg::P(inp.enter.cached_ptr()),
                Arg::P(inp.cn.cached_ptr()),
                Arg::P(inp.c_bias.cached_ptr()),
                Arg::P(inp.coef.cached_ptr()),
                Arg::P(inp.sn.cached_ptr()),
                Arg::P(inp.cs.cached_ptr()),
            ];
            if tiles {
                args.extend(ints(&[tt_i as i32]));
                let threads = 8 * (tt_i * rank).next_multiple_of(4);
                launch_runs(
                    &ctx,
                    &k.cb.inter_tiles,
                    cfg(
                        (b * inp.nc, nh, 64usize.div_ceil(tt_i)),
                        threads,
                        inter_smem,
                    ),
                    &args,
                    &[NO_PACKING],
                );
            } else {
                args.extend(ints(&[]));
                launch_runs(
                    &ctx,
                    &k.cb.inter,
                    cfg((b * inp.nc, nh, 64), 256, 0),
                    &args,
                    &[NO_PACKING],
                );
            }
            [bits_f32(&ctx, &dq), bits_f32(&ctx, &inter)]
        };
        assert!(
            inter_run(true) == inter_run(false),
            "{label}: inter-chunk term"
        );
        let dk0 = det(rows * nh * rn, 86, 1.0);
        let du0 = det(rows * nh * rank * hd, 87, 1.0);
        let (tq, kt, nk, scan_smem) = geom.scan_tiles;
        assert!(tq > 0, "{label}: no scan tile planned: {geom:?}");
        for dtype in DTYPES {
            let proj = typed_buf(&ctx, &det(rows * inp.ld_proj, 89, 1.0), dtype);
            // The forward scan: the gated outputs and the readouts.
            let scan_run = |tiles: bool| {
                let out = typed_buf(&ctx, &vec![0.0; rows * nh * hd], dtype);
                let y_out = GpuBuffer::zeros(s, rows * nh * rank * hd).unwrap();
                let mut args = vec![
                    Arg::P(out.cached_ptr()),
                    Arg::P(y_out.cached_ptr()),
                    Arg::P(proj.cached_ptr()),
                    Arg::L(inp.ld_proj as i64),
                    Arg::P(inp.bn.cached_ptr()),
                    Arg::P(inp.cn.cached_ptr()),
                    Arg::P(inp.coef.cached_ptr()),
                    Arg::P(inp.sn.cached_ptr()),
                    Arg::P(inp.cs.cached_ptr()),
                    Arg::P(inp.enter.cached_ptr()),
                    Arg::P(inp.b_bias.cached_ptr()),
                    Arg::P(inp.c_bias.cached_ptr()),
                    Arg::P(inp.d.cached_ptr()),
                    Arg::P(inp.mx.cached_ptr()),
                    Arg::P(inp.mz.cached_ptr()),
                    Arg::P(inp.mo.cached_ptr()),
                ];
                if tiles {
                    args.extend(ints(&[tq as i32, kt as i32, nk as i32]));
                    launch_runs(
                        &ctx,
                        k.chunk_scan_tiles.get(dtype),
                        chunk_scan_tiles_cfg((b, inp.nc, nh), (hd, rank), tq, scan_smem),
                        &args,
                        &[NO_PACKING],
                    );
                } else {
                    let (tt, st, smem) = geom.scan;
                    args.extend(ints(&[tt as i32, st as i32, geom.pt as i32]));
                    launch_yz_runs(
                        &ctx,
                        k.chunk_scan.get(dtype),
                        cfg(
                            (b * inp.nc, nh, 64usize.div_ceil(tt) * hd.div_ceil(geom.pt)),
                            (tt * geom.pt).next_multiple_of(32),
                            smem,
                        ),
                        &args,
                    );
                }
                (bits_typed(&ctx, &out), bits_f32(&ctx, &y_out))
            };
            assert!(
                scan_run(true) == scan_run(false),
                "{dtype:?} {label}: chunk scan"
            );
            let terms_run = |tiles: bool| {
                let (dk, du) = (f32_buf(&ctx, &dk0), f32_buf(&ctx, &du0));
                let dcs = GpuBuffer::zeros(s, rows * nh).unwrap();
                let mut args = vec![
                    Arg::P(dk.cached_ptr()),
                    Arg::P(du.cached_ptr()),
                    Arg::P(dcs.cached_ptr()),
                    Arg::P(inp.dst.cached_ptr()),
                    Arg::P(proj.cached_ptr()),
                    Arg::L(inp.ld_proj as i64),
                    Arg::P(inp.bn.cached_ptr()),
                    Arg::P(inp.b_bias.cached_ptr()),
                    Arg::P(inp.coef.cached_ptr()),
                    Arg::P(inp.sn.cached_ptr()),
                    Arg::P(inp.cs.cached_ptr()),
                    Arg::P(inp.mx.cached_ptr()),
                ];
                let scratch;
                if tiles {
                    args.extend(ints(&[tt_t as i32]));
                    let threads = 8 * (tt_t * rank).next_multiple_of(4);
                    launch_runs(
                        &ctx,
                        k.cb.state_terms_tiles.get(dtype),
                        cfg(
                            (b * inp.nc, nh, 64usize.div_ceil(tt_t)),
                            threads,
                            terms_smem,
                        ),
                        &args,
                        &[NO_PACKING],
                    );
                } else {
                    // Eight tokens a block, every row in global memory.
                    let tg = 8usize;
                    let blocks = b * inp.nc * nh * 64usize.div_ceil(tg);
                    scratch = GpuBuffer::zeros(s, blocks * (rn + rank * hd)).unwrap();
                    args.push(Arg::P(scratch.cached_ptr()));
                    args.extend(ints(&[tg as i32, 0]));
                    launch_runs(
                        &ctx,
                        k.cb.state_terms.get(dtype),
                        cfg((b * inp.nc, nh, 64usize.div_ceil(tg)), 256, 0),
                        &args,
                        &[NO_PACKING],
                    );
                }
                [
                    bits_f32(&ctx, &dk),
                    bits_f32(&ctx, &du),
                    bits_f32(&ctx, &dcs),
                ]
            };
            assert!(
                terms_run(true) == terms_run(false),
                "{dtype:?} {label}: state terms"
            );
            checked += 1;
        }
    }
    eprintln!("token tiles against the per-token kernels on {checked} cases");
}

/// The chunked backward's angle walk stages a batch of tokens at a time,
/// walks each angle's chain over them in time order, then sums each token's
/// products a ascending, so it gives the bits of the walk as it was
/// released, a token at a time: in f32, bf16 and f16, at windows that are
/// and are not whole batches, with one sequence a row and with rows packing
/// several, and from one angle to more than a lane holds.
#[test]
fn mimo_angle_walk_keeps_the_token_at_a_time_bits() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let k = Mamba3MimoKernels::compile(dev.context(), arch::arch0(), 1).unwrap();
    let frozen = frozen_module(
        dev.context(),
        arch::arch0(),
        64,
        include_str!("fixtures/m3m_angles_reference.cu"),
    );
    let ang_col = 3usize;
    let mut cases = 0;
    for (b, seq, nh, na, packed) in [
        (1usize, 1usize, 1usize, 1usize, false),
        (2, 15, 3, 2, false),
        (2, 16, 3, 2, true),
        (2, 17, 3, 7, true),
        (3, 100, 4, 32, true),
        (2, 129, 2, 64, false),
        (1, 70, 2, 128, true),
    ] {
        let rows = b * seq;
        let ld_proj = ang_col + na + 5;
        let dth = f32_buf(&ctx, &det(rows * nh * na, 0xA301, 1.0));
        let coef = f32_buf(&ctx, &det(rows * nh * 4, 0xA302, 1.0));
        // Sequences of 7 tokens a row, their ids as the kernel reads them.
        let ids: Vec<f32> = (0..rows)
            .map(|i| f32::from_bits(((i % seq) / 7) as u32))
            .collect();
        let ids = f32_buf(&ctx, &ids);
        let seq_idx = if packed { ids.cached_ptr() } else { 0 };
        for dtype in DTYPES {
            let proj = typed_buf(&ctx, &det(rows * ld_proj, 0xA303, 3.0), dtype);
            let run = |f: &CudaFunction, smem: usize, poison: f32| {
                let d_tanh = f32_buf(&ctx, &vec![poison; rows * nh * na]);
                let ddt = f32_buf(&ctx, &vec![poison; rows * nh]);
                let args = [
                    Arg::P(d_tanh.cached_ptr()),
                    Arg::P(ddt.cached_ptr()),
                    Arg::P(dth.cached_ptr()),
                    Arg::P(coef.cached_ptr()),
                    Arg::P(proj.cached_ptr()),
                    Arg::L(ld_proj as i64),
                    Arg::I(ang_col as i32),
                    Arg::I(seq as i32),
                    Arg::I(nh as i32),
                    Arg::I(na as i32),
                ];
                launch_runs(
                    &ctx,
                    f,
                    cfg((b, nh, 1), 32, smem),
                    &args,
                    &[Arg::P(seq_idx)],
                );
                (bits_f32(&ctx, &d_tanh), bits_f32(&ctx, &ddt))
            };
            let suffix = match dtype {
                WeightDtype::Bf16 => "_bf16",
                WeightDtype::F16 => "_f16",
                _ => "",
            };
            let released = frozen
                .load_function(&format!("m3mb_reference_angles{suffix}"))
                .unwrap();
            let want = run(&released, na * 4, 8192.0);
            let got = run(k.cb.angles.get(dtype), angle_walk_smem_bytes(na), -8192.0);
            let label = format!("{dtype:?} B {b} T {seq} nh {nh} na {na} packed {packed}");
            assert_eq!(want.0, got.0, "{label}: d_tanh_h");
            assert_eq!(want.1, got.1, "{label}: ddt_ang");
            cases += 1;
        }
    }
    eprintln!("m3mb_angles: the token-at-a-time walk's bits on {cases} cases");
}
