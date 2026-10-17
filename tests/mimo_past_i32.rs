//! Mamba-3 MIMO's chunked scan past 2^31 elements. Its kernels index with
//! 64 bits in a single build, so a window whose projection rows pass the
//! 32-bit range runs on the same kernels as a short one; this test launches
//! the chunked scan's kernels over a batch whose last sequence lies wholly
//! past the range and compares everything that sequence writes with the
//! same kernels over that sequence alone.
//!
//!   cargo test --release --features cuda --test mimo_past_i32 -- --ignored --nocapture
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use std::sync::Arc;

use cudarc::driver::sys::CUresult::CUDA_SUCCESS;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{y_runs, yz_runs};
use mamba_rs::mamba3_mimo::gpu::kernels::{
    ChunkedGeom, Mamba3MimoKernels, chunk_scan_tiles_cfg, chunked_geom,
};

const CHUNK: usize = 64;
const ACT: WeightDtype = WeightDtype::Bf16;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn floats(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut buf = GpuBuffer::zeros(&ctx.stream, data.len()).expect("alloc");
    ctx.stream.synchronize().expect("sync");
    buf.upload(&ctx.stream, data).expect("upload");
    buf
}

fn zeros(ctx: &GpuCtx, n: usize) -> GpuBuffer {
    GpuBuffer::zeros(&ctx.stream, n.max(1)).expect("alloc")
}

/// `bytes` bytes at `ptr`, read once the stream drains.
fn read(stream: &Arc<CudaStream>, ptr: u64, bytes: usize) -> Vec<u8> {
    stream.synchronize().expect("sync");
    let mut out = vec![0u8; bytes];
    let r = unsafe { cudarc::driver::sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), ptr, bytes) };
    assert_eq!(r, CUDA_SUCCESS, "read");
    out
}

#[derive(Clone, Copy)]
enum Arg {
    P(u64),
    I(i32),
    L(i64),
    F(f32),
}

fn launch(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    let mut b = ctx.stream.launch_builder(f);
    for a in args {
        match a {
            Arg::P(p) => b.arg(p),
            Arg::I(i) => b.arg(i),
            Arg::L(l) => b.arg(l),
            Arg::F(x) => b.arg(x),
        };
    }
    unsafe { b.launch(cfg) }.expect("launch");
}

/// [`launch`] once per run of the grid's y axis, each run's first block
/// appended, then `tail` (the packing, for the kernels that take one).
fn launch_runs(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg], tail: &[Arg]) {
    for (y_first, cfg) in y_runs(cfg) {
        let mut run = args.to_vec();
        run.push(Arg::I(y_first));
        run.extend_from_slice(tail);
        launch(ctx, f, cfg, &run);
    }
}

/// [`launch`] once per run of the grid's y and z axes, each run's first y
/// and z blocks appended, then the packing (none).
fn launch_yz_runs(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    for (y_first, z_first, cfg) in yz_runs(cfg) {
        let mut run = args.to_vec();
        run.extend([Arg::I(y_first), Arg::I(z_first), Arg::P(0)]);
        launch(ctx, f, cfg, &run);
    }
}

/// The scan's shape: `b` sequences of `t` tokens, `nh` heads of `hd`
/// channels, `ds` states, rank `r`, one group, `na` angles.
#[derive(Clone, Copy)]
struct Shape {
    b: usize,
    t: usize,
    nh: usize,
    hd: usize,
    ds: usize,
    r: usize,
    na: usize,
}

impl Shape {
    fn nc(&self) -> usize {
        self.t.div_ceil(CHUNK)
    }
    fn di(&self) -> usize {
        self.nh * self.hd
    }
    fn bc(&self) -> usize {
        self.r * self.ds
    }
    fn ld(&self) -> usize {
        2 * self.di() + 2 * self.bc() + 3 * self.nh + self.na
    }
}

/// One buffer of the scan: its name, its bytes per sequence, and whether a
/// run fills it from its inputs (`None`) or reads it seeded.
#[derive(Clone, Copy)]
enum Slot {
    Proj,
    Bn,
    Cn,
    Gated,
    Coef,
    Sn,
    Cs,
    E0,
    Contrib,
    Enter,
    Ssm,
    Kst,
    Vst,
    Ang,
}

const SLOTS: [Slot; 14] = [
    Slot::Proj,
    Slot::Bn,
    Slot::Cn,
    Slot::Gated,
    Slot::Coef,
    Slot::Sn,
    Slot::Cs,
    Slot::E0,
    Slot::Contrib,
    Slot::Enter,
    Slot::Ssm,
    Slot::Kst,
    Slot::Vst,
    Slot::Ang,
];

impl Slot {
    /// Bytes per sequence.
    fn bytes(self, s: Shape) -> usize {
        let a = ACT.size_bytes();
        let f = 4;
        let rows = s.t;
        match self {
            Slot::Proj => rows * s.ld() * a,
            Slot::Bn | Slot::Cn => rows * s.bc() * f,
            Slot::Gated => rows * s.di() * a,
            Slot::Coef => rows * s.nh * 4 * f,
            Slot::Sn | Slot::Cs => rows * s.nh * s.na * f,
            Slot::E0 => s.nh * f,
            Slot::Contrib | Slot::Enter => s.nc() * s.nh * s.hd * s.ds * f,
            Slot::Ssm => s.nh * s.hd * s.ds * f,
            Slot::Kst => s.nh * s.r * s.ds * f,
            Slot::Vst => s.nh * s.hd * f,
            Slot::Ang => s.nh * s.na * f,
        }
    }
    /// The inputs a run reads: the projection, the normed B and C and the
    /// carried state; every other slot is the run's output.
    fn seeded(self) -> bool {
        matches!(
            self,
            Slot::Proj | Slot::Bn | Slot::Cn | Slot::Ssm | Slot::Kst | Slot::Vst | Slot::Ang
        )
    }
    fn name(self) -> &'static str {
        match self {
            Slot::Proj => "proj",
            Slot::Bn => "bn",
            Slot::Cn => "cn",
            Slot::Gated => "gated",
            Slot::Coef => "coef",
            Slot::Sn => "sn",
            Slot::Cs => "cs",
            Slot::E0 => "e0",
            Slot::Contrib => "contrib",
            Slot::Enter => "enter",
            Slot::Ssm => "ssm",
            Slot::Kst => "kst",
            Slot::Vst => "vst",
            Slot::Ang => "ang",
        }
    }
}

/// Device buffers of every slot for `s.b` sequences, as byte pools.
struct Pools {
    bufs: Vec<GpuBuffer>,
}

impl Pools {
    fn new(ctx: &GpuCtx, s: Shape) -> Self {
        let bufs = SLOTS
            .iter()
            .map(|slot| zeros(ctx, (s.b * slot.bytes(s)).div_ceil(4)))
            .collect();
        Self { bufs }
    }
    /// Sequence `q`'s part of `slot`.
    fn at(&self, s: Shape, slot: Slot, q: usize) -> u64 {
        let i = SLOTS
            .iter()
            .position(|x| x.name() == slot.name())
            .expect("slot");
        self.bufs[i].cached_ptr() + (q * slot.bytes(s)) as u64
    }
}

/// The layer weights the scan reads.
struct Weights {
    dt_bias: GpuBuffer,
    b_bias: GpuBuffer,
    c_bias: GpuBuffer,
    d: GpuBuffer,
    mimo_x: GpuBuffer,
    mimo_z: GpuBuffer,
    mimo_o: GpuBuffer,
}

/// Every kernel of the chunked scan, in the order and with the geometry the
/// layer launches them.
fn run(ctx: &GpuCtx, k: &Mamba3MimoKernels, geom: &ChunkedGeom, s: Shape, p: &Pools, w: &Weights) {
    let at = |slot| p.at(s, slot, 0);
    let (b, nc) = (s.b, s.nc());
    let ld = Arg::L(s.ld() as i64);
    let (seq, nh, ng, hd, ds, r, na) = (
        Arg::I(s.t as i32),
        Arg::I(s.nh as i32),
        Arg::I(1),
        Arg::I(s.hd as i32),
        Arg::I(s.ds as i32),
        Arg::I(s.r as i32),
        Arg::I(s.na as i32),
    );
    let heads = 2 * s.di() + 2 * s.bc();
    let ang_col = heads + 3 * s.nh;
    launch_runs(
        ctx,
        k.chunk_prep.get(ACT),
        LaunchConfig {
            grid_dim: (b as u32, s.nh as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: 0,
        },
        &[
            Arg::P(at(Slot::Coef)),
            Arg::P(at(Slot::Sn)),
            Arg::P(at(Slot::Cs)),
            Arg::P(at(Slot::E0)),
            Arg::P(at(Slot::Ang)),
            Arg::P(at(Slot::Proj)),
            ld,
            Arg::P(w.dt_bias.cached_ptr()),
            seq,
            nh,
            Arg::I(heads as i32),
            Arg::I(ang_col as i32),
            na,
            Arg::F(1e-4),
        ],
        &[Arg::P(0)],
    );
    let state_in = [
        Arg::P(at(Slot::Contrib)),
        Arg::P(at(Slot::Proj)),
        ld,
        Arg::P(at(Slot::Bn)),
        Arg::P(at(Slot::Coef)),
        Arg::P(at(Slot::Sn)),
        Arg::P(at(Slot::Cs)),
        Arg::P(w.b_bias.cached_ptr()),
        Arg::P(w.mimo_x.cached_ptr()),
        seq,
        nh,
        ng,
        hd,
        ds,
        r,
        na,
    ];
    if geom.state_tiles.0 > 0 {
        let (jc, sl, pg, smem) = geom.state_tiles;
        let mut args = state_in.to_vec();
        args.extend([Arg::I(jc as i32), Arg::I(sl as i32), Arg::I(pg as i32)]);
        launch_yz_runs(
            ctx,
            k.chunk_state_tiles.get(ACT),
            LaunchConfig {
                grid_dim: (
                    (b * nc) as u32,
                    s.nh as u32,
                    (s.ds.div_ceil(32 * jc) * s.hd.div_ceil(pg)) as u32,
                ),
                block_dim: ((8 * pg) as u32, 1, 1),
                shared_mem_bytes: smem as u32,
            },
            &args,
        );
    } else {
        let (nt, sl, smem) = geom.state;
        let mut args = state_in.to_vec();
        args.extend([Arg::I(nt as i32), Arg::I(sl as i32), Arg::I(geom.pt as i32)]);
        launch_yz_runs(
            ctx,
            k.chunk_state.get(ACT),
            LaunchConfig {
                grid_dim: (
                    (b * nc) as u32,
                    s.nh as u32,
                    (s.ds.div_ceil(nt) * s.hd.div_ceil(geom.pt)) as u32,
                ),
                block_dim: ((nt * geom.pt) as u32, 1, 1),
                shared_mem_bytes: smem as u32,
            },
            &args,
        );
    }
    launch_runs(
        ctx,
        &k.state_pass,
        LaunchConfig {
            grid_dim: ((b * s.nh) as u32, (s.hd * s.ds).div_ceil(256) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        },
        &[
            Arg::P(at(Slot::Enter)),
            Arg::P(at(Slot::Ssm)),
            Arg::P(at(Slot::Contrib)),
            Arg::P(at(Slot::Coef)),
            Arg::P(at(Slot::E0)),
            Arg::P(at(Slot::Kst)),
            Arg::P(at(Slot::Vst)),
            Arg::P(w.mimo_x.cached_ptr()),
            Arg::I(b as i32),
            seq,
            nh,
            hd,
            ds,
            r,
        ],
        &[Arg::P(0)],
    );
    let scan_in = [
        Arg::P(at(Slot::Gated)),
        Arg::P(0),
        Arg::P(at(Slot::Proj)),
        ld,
        Arg::P(at(Slot::Bn)),
        Arg::P(at(Slot::Cn)),
        Arg::P(at(Slot::Coef)),
        Arg::P(at(Slot::Sn)),
        Arg::P(at(Slot::Cs)),
        Arg::P(at(Slot::Enter)),
        Arg::P(w.b_bias.cached_ptr()),
        Arg::P(w.c_bias.cached_ptr()),
        Arg::P(w.d.cached_ptr()),
        Arg::P(w.mimo_x.cached_ptr()),
        Arg::P(w.mimo_z.cached_ptr()),
        Arg::P(w.mimo_o.cached_ptr()),
        seq,
        nh,
        ng,
        hd,
        ds,
        r,
        na,
    ];
    if geom.scan_tiles.0 > 0 {
        let (tq, kt, nk, smem) = geom.scan_tiles;
        let mut args = scan_in.to_vec();
        args.extend([Arg::I(tq as i32), Arg::I(kt as i32), Arg::I(nk as i32)]);
        launch_runs(
            ctx,
            k.chunk_scan_tiles_for(s.hd).get(ACT),
            chunk_scan_tiles_cfg((b, nc, s.nh), (s.hd, s.r), tq, smem),
            &args,
            &[Arg::P(0)],
        );
    } else {
        let (tt, st, smem) = geom.scan;
        let mut args = scan_in.to_vec();
        args.extend([Arg::I(tt as i32), Arg::I(st as i32), Arg::I(geom.pt as i32)]);
        launch_yz_runs(
            ctx,
            k.chunk_scan.get(ACT),
            LaunchConfig {
                grid_dim: (
                    (b * nc) as u32,
                    s.nh as u32,
                    (CHUNK.div_ceil(tt) * s.hd.div_ceil(geom.pt)) as u32,
                ),
                block_dim: ((tt * geom.pt).next_multiple_of(32) as u32, 1, 1),
                shared_mem_bytes: smem as u32,
            },
            &args,
        );
    }
    launch_runs(
        ctx,
        k.chunk_writeback.get(ACT),
        LaunchConfig {
            grid_dim: (b as u32, s.nh as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        },
        &[
            Arg::P(at(Slot::Kst)),
            Arg::P(at(Slot::Vst)),
            Arg::P(at(Slot::Proj)),
            ld,
            Arg::P(at(Slot::Bn)),
            Arg::P(at(Slot::Sn)),
            Arg::P(at(Slot::Cs)),
            Arg::P(w.b_bias.cached_ptr()),
            seq,
            nh,
            ng,
            hd,
            ds,
            r,
            na,
        ],
        &[],
    );
}

/// Past 2^31 elements of the projection rows the chunked scan writes the
/// last sequence of a batch, whose rows lie wholly past the range, what it
/// writes that sequence alone: the step scalars, the angles, every chunk's
/// contribution and entering state, the gated outputs and the carried
/// state.
#[test]
#[ignore = "20 GiB of projection and B and C rows; run on a 32 GB card"]
fn the_chunked_scan_gives_a_sequence_past_i32_max_its_own_bits() {
    let s = Shape {
        b: 4,
        t: 1_304_000,
        nh: 1,
        hd: 16,
        ds: 64,
        r: 4,
        na: 2,
    };
    let one = Shape { b: 1, ..s };
    let last = s.b - 1;
    assert!(last * s.t * s.ld() > i32::MAX as usize);
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let k = Mamba3MimoKernels::compile(ctx.stream.context(), arch::arch0(), s.r).expect("MIMO");
    let geom = chunked_geom((s.hd, s.ds, s.r), k.smem_optin, false, k.chunk_threads)
        .expect("a chunked scan of this head fits");

    let w = Weights {
        dt_bias: floats(&ctx, &det(s.nh, 1, 0.5)),
        b_bias: floats(&ctx, &det(s.nh * s.r * s.ds, 2, 0.5)),
        c_bias: floats(&ctx, &det(s.nh * s.r * s.ds, 3, 0.5)),
        d: floats(&ctx, &det(s.nh, 4, 1.0)),
        mimo_x: floats(&ctx, &det(s.nh * s.r * s.hd, 5, 0.5)),
        mimo_z: floats(&ctx, &det(s.nh * s.r * s.hd, 6, 0.5)),
        mimo_o: floats(&ctx, &det(s.nh * s.r * s.hd, 7, 0.5)),
    };
    // Every sequence's inputs come from one seeded sequence, the last one's
    // from another, so a row read at a wrapped offset reads other bits.
    let seed_seq = |seed: u32| -> Vec<(Slot, Vec<u8>)> {
        SLOTS
            .iter()
            .filter(|slot| slot.seeded())
            .map(|&slot| {
                let n = slot.bytes(one);
                let bytes = if matches!(slot, Slot::Proj) {
                    let vals = det(n / ACT.size_bytes(), seed, 1.0);
                    let buf = DtypedBuf::zeros(st, vals.len(), ACT).expect("alloc");
                    buf.upload_f32(st, &vals).expect("upload");
                    read(st, buf.cached_ptr(), n)
                } else {
                    det(n / 4, seed + 1, 0.5)
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect()
                };
                (slot, bytes)
            })
            .collect()
    };
    let others = seed_seq(10);
    let own_in = seed_seq(20);
    let batch = Pools::new(&ctx, s);
    let alone = Pools::new(&ctx, one);
    let put = |pools: &Pools, shape: Shape, q: usize, data: &[(Slot, Vec<u8>)]| {
        st.synchronize().expect("sync");
        for (slot, bytes) in data {
            let r = unsafe {
                cudarc::driver::sys::cuMemcpyHtoD_v2(
                    pools.at(shape, *slot, q),
                    bytes.as_ptr().cast(),
                    bytes.len(),
                )
            };
            assert_eq!(r, CUDA_SUCCESS, "upload {}", slot.name());
        }
    };
    for q in 0..last {
        put(&batch, s, q, &others);
    }
    put(&batch, s, last, &own_in);
    put(&alone, one, 0, &own_in);

    run(&ctx, &k, &geom, s, &batch, &w);
    run(&ctx, &k, &geom, one, &alone, &w);
    for slot in SLOTS {
        let n = slot.bytes(one);
        let a = read(st, batch.at(s, slot, last), n);
        let b = read(st, alone.at(one, slot, 0), n);
        if let Some(i) = a.iter().zip(&b).position(|(x, y)| x != y) {
            panic!("{}: byte {i} of {n} differs", slot.name());
        }
    }
}
