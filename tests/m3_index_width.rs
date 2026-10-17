//! The two builds of the Mamba-3 SISO kernels. Each is compiled with `int`
//! offsets and with `long long` offsets (the `_i64` entry), and a window
//! takes the `int` builds while every tensor it indexes fits the 32-bit
//! range. The two builds give the same bits on every indexing pattern the
//! family has (a flat walk over the projection, the walk of a sequence over
//! its window, the chunk grid, and the angle gradients with their axis-0
//! reduction), and past 2^31 the `_i64` builds give every row, sequence and
//! chunk exactly what the `int` build gives it alone. The qualification
//! runs prove the bits over every recorded case with
//! `MAMBA_RS_INDEX_WIDTH=i64`.
//!
//!   cargo test --release --features cuda --test m3_index_width -- --include-ignored --nocapture
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use std::sync::Arc;

use cudarc::driver::sys::CUresult::CUDA_SUCCESS;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::launch::{grid_1d_cover, y_runs};
use mamba_rs::mamba3_siso::gpu::kernels::{
    ChunkArm, Mamba3Kernels, chunk_scan_cfg, chunk_scan_coop_cfg, chunk_scan_sliced_cfg,
};

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
    GpuBuffer::zeros(&ctx.stream, n).expect("alloc")
}

/// `n` floats at `ptr`, read once the stream drains.
fn read(stream: &Arc<CudaStream>, ptr: u64, n: usize) -> Vec<u32> {
    stream.synchronize().expect("sync");
    let mut out = vec![0u32; n];
    let r = unsafe { cudarc::driver::sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), ptr, n * 4) };
    assert_eq!(r, CUDA_SUCCESS, "read");
    out
}

/// Copy `n` floats from `src` to `dst` on the stream.
fn copy(stream: &Arc<CudaStream>, dst: u64, src: u64, n: usize) {
    let r =
        unsafe { cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, n * 4, stream.cu_stream()) };
    assert_eq!(r, CUDA_SUCCESS, "copy");
}

/// Fill `n` floats at `ptr` with the bits `word`.
fn fill(stream: &Arc<CudaStream>, ptr: u64, word: u32, n: usize) {
    let r = unsafe { cudarc::driver::sys::cuMemsetD32Async(ptr, word, n, stream.cu_stream()) };
    assert_eq!(r, CUDA_SUCCESS, "memset");
}

/// `ptr` moved past `n` floats.
fn at(ptr: u64, n: usize) -> u64 {
    ptr + (n * 4) as u64
}

/// A kernel argument.
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
/// appended as the kernel's last argument.
fn launch_runs(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    for (y_first, cfg) in y_runs(cfg) {
        let mut run = args.to_vec();
        // The run's first y block, then the packing every kernel here takes
        // last (none).
        run.extend([Arg::I(y_first), Arg::P(0)]);
        launch(ctx, f, cfg, &run);
    }
}

fn compile(ctx: &GpuCtx, state_cap: usize) -> Mamba3Kernels {
    Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), state_cap)
        .expect("M3 kernels")
}

/// The split's shape: `rows` rows of the projection.
#[derive(Clone, Copy)]
struct Split {
    rows: usize,
    di: usize,
    ng: usize,
    ds: usize,
    nh: usize,
    na: usize,
}

impl Split {
    fn width(&self) -> usize {
        2 * self.di + 2 * self.ng * self.ds + 3 * self.nh + self.na
    }
}

/// The split's buffers, every output of one row next to the row's offset.
struct SplitIo {
    proj: u64,
    bias: u64,
    z: u64,
    x: u64,
    b: u64,
    c: u64,
    dt: u64,
    a: u64,
    trap: u64,
    angles: u64,
    dt_raw: u64,
    a_raw: u64,
    trap_raw: u64,
}

impl SplitIo {
    /// The outputs of `rows` rows from row `r`, as (name, floats) pairs.
    fn outputs(&self, s: Split, r: usize, rows: usize) -> Vec<(&'static str, u64, usize)> {
        let mut out = vec![
            ("z", at(self.z, r * s.di), rows * s.di),
            ("x", at(self.x, r * s.di), rows * s.di),
            ("dt", at(self.dt, r * s.nh), rows * s.nh),
            ("a", at(self.a, r * s.nh), rows * s.nh),
            ("trap", at(self.trap, r * s.nh), rows * s.nh),
            ("angles", at(self.angles, r * s.na), rows * s.na),
            ("dt_raw", at(self.dt_raw, r * s.nh), rows * s.nh),
            ("a_raw", at(self.a_raw, r * s.nh), rows * s.nh),
            ("trap_raw", at(self.trap_raw, r * s.nh), rows * s.nh),
        ];
        if self.b != 0 {
            out.push(("B", at(self.b, r * s.ng * s.ds), rows * s.ng * s.ds));
            out.push(("C", at(self.c, r * s.ng * s.ds), rows * s.ng * s.ds));
        }
        out
    }
}

fn run_split(ctx: &GpuCtx, f: &CudaFunction, s: Split, io: &SplitIo) {
    let p = |v: u64| Arg::P(v);
    launch(
        ctx,
        f,
        grid_1d_cover(s.rows * s.width()).expect("grid"),
        &[
            p(io.z),
            p(io.x),
            p(io.b),
            p(io.c),
            p(io.dt),
            p(io.a),
            p(io.trap),
            p(io.angles),
            p(io.dt_raw),
            p(io.a_raw),
            p(io.trap_raw),
            p(io.proj),
            p(io.bias),
            Arg::F(1e-4),
            Arg::I(s.rows as i32),
            Arg::I(s.di as i32),
            Arg::I(s.ng as i32),
            Arg::I(s.ds as i32),
            Arg::I(s.nh as i32),
            Arg::I(s.na as i32),
        ],
    );
}

fn split_io(ctx: &GpuCtx, s: Split, proj: u64, with_bc: bool) -> (SplitIo, Vec<GpuBuffer>) {
    let bufs: Vec<GpuBuffer> = [
        s.nh,
        s.rows * s.di,
        s.rows * s.di,
        if with_bc { s.rows * s.ng * s.ds } else { 1 },
        if with_bc { s.rows * s.ng * s.ds } else { 1 },
        s.rows * s.nh,
        s.rows * s.nh,
        s.rows * s.nh,
        s.rows * s.na,
        s.rows * s.nh,
        s.rows * s.nh,
        s.rows * s.nh,
    ]
    .iter()
    .enumerate()
    .map(|(i, &n)| {
        if i == 0 {
            floats(ctx, &det(n, 7, 1.0))
        } else {
            zeros(ctx, n)
        }
    })
    .collect();
    let ptr = |i: usize| bufs[i].cached_ptr();
    let io = SplitIo {
        proj,
        bias: ptr(0),
        z: ptr(1),
        x: ptr(2),
        b: if with_bc { ptr(3) } else { 0 },
        c: if with_bc { ptr(4) } else { 0 },
        dt: ptr(5),
        a: ptr(6),
        trap: ptr(7),
        angles: ptr(8),
        dt_raw: ptr(9),
        a_raw: ptr(10),
        trap_raw: ptr(11),
    };
    (io, bufs)
}

/// A sequential scan's shape on the warp tier: one warp per head.
#[derive(Clone, Copy)]
struct Seq {
    b: usize,
    t: usize,
    nh: usize,
    hd: usize,
    ds: usize,
}

impl Seq {
    fn di(&self) -> usize {
        self.nh * self.hd
    }
    fn tape(&self) -> usize {
        (self.t + 1) * self.di() * self.ds
    }
    fn cfg(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.b as u32, self.nh as u32, 1),
            block_dim: (self.hd as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }
    fn ints(&self) -> [Arg; 5] {
        [
            Arg::I(self.b as i32),
            Arg::I(self.t as i32),
            Arg::I(self.nh as i32),
            Arg::I(self.hd as i32),
            Arg::I(self.ds as i32),
        ]
    }
}

/// Every buffer of the sequential forward and backward, per sequence:
/// (name, floats per sequence, input seed or none for an output).
const SEQ_BUFS: [(&str, Per, Option<u32>); 23] = [
    ("ssm", Per::State, Some(11)),
    ("k_state", Per::KState, Some(12)),
    ("v_state", Per::VState, Some(13)),
    ("y", Per::Rows, None),
    ("h_saved", Per::Tape, None),
    ("k_prev", Per::KRows, None),
    ("v_prev", Per::Rows, None),
    ("x", Per::Rows, Some(14)),
    ("k", Per::KRows, Some(15)),
    ("q", Per::KRows, Some(16)),
    ("alpha", Per::Heads, Some(17)),
    ("beta", Per::Heads, Some(18)),
    ("gamma", Per::Heads, Some(19)),
    ("d", Per::Shared, Some(20)),
    ("dy", Per::Rows, Some(21)),
    ("dx", Per::Rows, None),
    ("dk", Per::KRows, None),
    ("dq", Per::KRows, None),
    ("dalpha", Per::Heads, None),
    ("dbeta", Per::Heads, None),
    ("dgamma", Per::Heads, None),
    ("dd_local", Per::Inner, None),
    ("unused", Per::Shared, None),
];

#[derive(Clone, Copy)]
enum Per {
    State,
    KState,
    VState,
    Rows,
    KRows,
    Heads,
    Tape,
    Inner,
    Shared,
}

impl Per {
    fn len(self, s: Seq) -> usize {
        match self {
            Per::State => s.di() * s.ds,
            Per::KState => s.nh * s.ds,
            Per::VState => s.di(),
            Per::Rows => s.t * s.di(),
            Per::KRows => s.t * s.nh * s.ds,
            Per::Heads => s.t * s.nh,
            Per::Tape => s.tape(),
            Per::Inner => s.di(),
            Per::Shared => 0,
        }
    }
}

/// The scan's buffers for `s.b` sequences; a shared buffer holds `nh` floats.
fn seq_bufs(ctx: &GpuCtx, s: Seq) -> Vec<GpuBuffer> {
    SEQ_BUFS
        .iter()
        .map(|&(_, per, seed)| {
            let n = match per {
                Per::Shared => s.nh,
                _ => s.b * per.len(s),
            };
            match seed {
                // The decays stay in (0, 1) so the walk stays finite.
                Some(sd) if matches!(per, Per::Heads) && sd == 17 => {
                    let v: Vec<f32> = det(n, sd, 0.4).iter().map(|v| 0.55 + v).collect();
                    floats(ctx, &v)
                }
                Some(sd) => floats(ctx, &det(n, sd, 0.5)),
                None => zeros(ctx, n),
            }
        })
        .collect()
}

/// Pointers to sequence `q`'s slice of every buffer.
fn seq_ptrs(bufs: &[GpuBuffer], s: Seq, q: usize) -> Vec<u64> {
    SEQ_BUFS
        .iter()
        .zip(bufs)
        .map(|(&(_, per, _), buf)| match per {
            Per::Shared => buf.cached_ptr(),
            _ => at(buf.cached_ptr(), q * per.len(s)),
        })
        .collect()
}

fn run_seq(ctx: &GpuCtx, fwd: &CudaFunction, bwd: &CudaFunction, s: Seq, p: &[u64]) {
    let ptr = |name: &str| {
        Arg::P(
            p[SEQ_BUFS
                .iter()
                .position(|(n, _, _)| *n == name)
                .expect(name)],
        )
    };
    let fwd_args: Vec<Arg> = [
        "ssm", "k_state", "v_state", "y", "h_saved", "k_prev", "v_prev", "x", "k", "q", "alpha",
        "beta", "gamma", "d",
    ]
    .iter()
    .map(|n| ptr(n))
    .chain(s.ints())
    .collect();
    launch_runs(ctx, fwd, s.cfg(), &fwd_args);
    let bwd_args: Vec<Arg> = [
        "h_saved", "k_prev", "v_prev", "x", "k", "q", "alpha", "beta", "gamma", "d", "dy", "dx",
        "dk", "dq", "dalpha", "dbeta", "dgamma", "dd_local",
    ]
    .iter()
    .map(|n| ptr(n))
    .chain(s.ints())
    .collect();
    launch_runs(ctx, bwd, s.cfg(), &bwd_args);
}

/// The chunked scan's shape.
#[derive(Clone, Copy)]
struct Chunked {
    b: usize,
    t: usize,
    nh: usize,
    hd: usize,
    ds: usize,
}

const CS: usize = 64;

impl Chunked {
    fn nc(&self) -> usize {
        self.t.div_ceil(CS)
    }
    fn ints(&self) -> [Arg; 6] {
        [
            Arg::I(self.b as i32),
            Arg::I(self.t as i32),
            Arg::I(self.nh as i32),
            Arg::I(self.hd as i32),
            Arg::I(self.ds as i32),
            Arg::I(CS as i32),
        ]
    }
}

/// The chunked scan's buffers: y, x, Q, K_scaled, qk_dot, dA_cumsum,
/// entering states, D, with their floats per time step or per chunk.
struct ChunkIo {
    y: u64,
    x: u64,
    q: u64,
    k: u64,
    qk: u64,
    da: u64,
    states: u64,
    d: u64,
}

impl ChunkIo {
    /// The buffers from chunk `c` on.
    fn past_chunks(&self, s: Chunked, c: usize) -> Self {
        let t0 = c * CS;
        let di = s.nh * s.hd;
        Self {
            y: at(self.y, t0 * di),
            x: at(self.x, t0 * di),
            q: at(self.q, t0 * s.nh * s.ds),
            k: at(self.k, t0 * s.nh * s.ds),
            qk: at(self.qk, t0 * s.nh),
            da: at(self.da, c * s.nh * CS),
            states: at(self.states, c * s.nh * s.hd * s.ds),
            d: self.d,
        }
    }

    fn args(&self, s: Chunked) -> Vec<Arg> {
        [
            self.y,
            self.x,
            self.q,
            self.k,
            self.qk,
            self.da,
            self.states,
            self.d,
        ]
        .into_iter()
        .map(Arg::P)
        .chain(s.ints())
        .collect()
    }
}

/// Decay cumsums of every chunk of `s`, nonpositive and falling inside a
/// chunk, as the chunked forward writes them.
fn decay_cumsums(s: Chunked, seed: u32) -> Vec<f32> {
    let steps = det(s.b * s.nc() * s.nh * CS, seed, 0.05);
    let mut out = vec![0.0f32; steps.len()];
    for (row, src) in out.chunks_mut(CS).zip(steps.chunks(CS)) {
        let mut acc = 0.0f32;
        for (o, v) in row.iter_mut().zip(src) {
            acc -= v.abs();
            *o = acc;
        }
    }
    out
}

fn chunk_bufs(ctx: &GpuCtx, s: Chunked) -> (ChunkIo, Vec<GpuBuffer>) {
    let di = s.nh * s.hd;
    let rows = s.b * s.t;
    let bufs = vec![
        zeros(ctx, rows * di),
        floats(ctx, &det(rows * di, 31, 0.5)),
        floats(ctx, &det(rows * s.nh * s.ds, 32, 0.5)),
        floats(ctx, &det(rows * s.nh * s.ds, 33, 0.5)),
        floats(ctx, &det(rows * s.nh, 34, 0.5)),
        floats(ctx, &decay_cumsums(s, 35)),
        floats(ctx, &det(s.b * s.nc() * s.nh * s.hd * s.ds, 36, 0.5)),
        floats(ctx, &det(s.nh, 37, 1.0)),
    ];
    let p = |i: usize| bufs[i].cached_ptr();
    let io = ChunkIo {
        y: p(0),
        x: p(1),
        q: p(2),
        k: p(3),
        qk: p(4),
        da: p(5),
        states: p(6),
        d: p(7),
    };
    (io, bufs)
}

/// The f32 builds of the intra-chunk scan an arm runs, int then `_i64`.
fn scan_builds(m3k: &Mamba3Kernels, arm: ChunkArm) -> [&CudaFunction; 2] {
    match arm {
        ChunkArm::Whole => [
            &m3k.m3_chunk_scan_fwd_coop.i32,
            &m3k.m3_chunk_scan_fwd_coop.i64,
        ],
        ChunkArm::Sliced => [
            &m3k.m3_chunk_scan_fwd_sliced_typed.i32.f32,
            &m3k.m3_chunk_scan_fwd_sliced_typed.i64.f32,
        ],
    }
}

fn assert_bits(what: &str, a: &[u32], b: &[u32]) {
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!(
            "{what}: element {i} of {} differs: {:#010x} != {:#010x}",
            a.len(),
            a[i],
            b[i]
        );
    }
    assert_eq!(a.len(), b.len(), "{what}: lengths");
}

/// The int and `_i64` builds of the split, the sequential scans, the chunk
/// scan (both the cooperative and the fallback kernel), the angle backward
/// and the axis-0 reduction against its wide form give the same bits on the
/// same inputs.
#[test]
fn both_builds_of_the_siso_kernels_give_the_same_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let m3k = compile(&ctx, 16);

    // The split: every column class, B and C written.
    let s = Split {
        rows: 37,
        di: 32,
        ng: 1,
        ds: 16,
        nh: 2,
        na: 4,
    };
    let proj = floats(&ctx, &det(s.rows * s.width(), 3, 2.0));
    let outs: Vec<Vec<(&str, Vec<u32>)>> = [&m3k.m3_split.i32, &m3k.m3_split.i64]
        .iter()
        .map(|f| {
            let (io, _bufs) = split_io(&ctx, s, proj.cached_ptr(), true);
            run_split(&ctx, f, s, &io);
            io.outputs(s, 0, s.rows)
                .into_iter()
                .map(|(name, ptr, n)| (name, read(st, ptr, n)))
                .collect()
        })
        .collect();
    for ((name, a), (_, b)) in outs[0].iter().zip(&outs[1]) {
        assert_bits(&format!("split {name}"), a, b);
    }

    // The sequential forward and backward, general and by state width.
    let q = Seq {
        b: 2,
        t: 40,
        nh: 2,
        hd: 16,
        ds: 16,
    };
    let state = m3k
        .m3_burnin_fwd_by_state
        .iter()
        .find(|(w, _)| *w == q.ds)
        .map(|(_, f)| f)
        .expect("ds16");
    let state_bwd = m3k
        .m3_backward_seq_by_state
        .iter()
        .find(|(w, _)| *w == q.ds)
        .map(|(_, f)| f)
        .expect("ds16");
    for (fwd, bwd, what) in [
        (&m3k.m3_burnin_fwd, &m3k.m3_backward_seq, "general"),
        (state, state_bwd, "by state"),
    ] {
        let results: Vec<Vec<Vec<u32>>> = [(&fwd.i32, &bwd.i32), (&fwd.i64, &bwd.i64)]
            .iter()
            .map(|(f, b)| {
                let bufs = seq_bufs(&ctx, q);
                run_seq(&ctx, f, b, q, &seq_ptrs(&bufs, q, 0));
                bufs.iter()
                    .map(|buf| read(st, buf.cached_ptr(), buf.len()))
                    .collect()
            })
            .collect();
        for (i, (a, b)) in results[0].iter().zip(&results[1]).enumerate() {
            assert_bits(&format!("sequential {what} {}", SEQ_BUFS[i].0), a, b);
        }
    }

    // The chunk scan: three chunks, the last one short.
    let c = Chunked {
        b: 2,
        t: 130,
        nh: 3,
        hd: 16,
        ds: 16,
    };
    let coop_cfg = chunk_scan_coop_cfg(c.b, c.nc(), c.nh, c.hd, c.ds, CS)
        .expect("the small chunk scan fits the cooperative tile");
    let sliced_cfg = chunk_scan_sliced_cfg(c.b, c.nc(), c.nh, c.hd);
    for (arm, cfg, what) in [
        (ChunkArm::Whole, coop_cfg, "cooperative"),
        (ChunkArm::Sliced, sliced_cfg, "sliced"),
    ] {
        let ys: Vec<Vec<u32>> = scan_builds(&m3k, arm)
            .iter()
            .map(|f| {
                let (io, bufs) = chunk_bufs(&ctx, c);
                launch_runs(&ctx, f, cfg, &io.args(c));
                read(st, io.y, bufs[0].len())
            })
            .collect();
        assert_bits(&format!("chunk scan {what}"), &ys[0], &ys[1]);
    }

    // The angle backward and both forms of its axis-0 reductions.
    let (b, t, nh, na) = (2usize, 70usize, 3usize, 4usize);
    let d_cumsum = floats(&ctx, &det(b * t * nh * na, 41, 0.5));
    let raw = floats(&ctx, &det(b * t * na, 42, 1.0));
    let dt = floats(&ctx, &det(b * t * nh, 43, 0.5));
    let parts: Vec<Vec<u32>> = [&m3k.m3_angle_dt_bwd_seq.i32, &m3k.m3_angle_dt_bwd_seq.i64]
        .iter()
        .map(|f| {
            let partials = zeros(&ctx, 2 * b * t * nh * na);
            let dt_part = at(partials.cached_ptr(), b * t * nh * na);
            let cfg = LaunchConfig {
                grid_dim: (b as u32, (nh * na).div_ceil(256) as u32, 1),
                block_dim: ((nh * na).min(256) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            launch_runs(
                &ctx,
                f,
                cfg,
                &[
                    Arg::P(partials.cached_ptr()),
                    Arg::P(dt_part),
                    Arg::P(d_cumsum.cached_ptr()),
                    Arg::P(raw.cached_ptr()),
                    Arg::P(dt.cached_ptr()),
                    Arg::I(b as i32),
                    Arg::I(t as i32),
                    Arg::I(nh as i32),
                    Arg::I(na as i32),
                ],
            );
            read(st, partials.cached_ptr(), partials.len())
        })
        .collect();
    assert_bits("angle backward partials", &parts[0], &parts[1]);

    // The angle gradients' reductions: nh rows of B*T*na columns and na
    // rows of B*T*nh, the wide form in two launches from a column offset.
    let partials = floats(&ctx, &det(nh * b * t * na, 44, 0.5));
    for (rows, cols) in [(nh, b * t * na), (na, b * t * nh)] {
        let lanes = rows.next_power_of_two().clamp(32, 256) as u32;
        let narrow = zeros(&ctx, cols);
        launch(
            &ctx,
            &m3k.reduce_sum_axis0,
            LaunchConfig {
                grid_dim: (cols as u32, 1, 1),
                block_dim: (lanes, 1, 1),
                shared_mem_bytes: lanes * 4,
            },
            &[
                Arg::P(narrow.cached_ptr()),
                Arg::P(partials.cached_ptr()),
                Arg::I(rows as i32),
                Arg::I(cols as i32),
                Arg::I(0),
            ],
        );
        let wide = zeros(&ctx, cols);
        let first = cols / 3;
        for (col0, n) in [(0, first), (first, cols - first)] {
            launch(
                &ctx,
                &m3k.reduce_sum_axis0_wide,
                LaunchConfig {
                    grid_dim: (n as u32, 1, 1),
                    block_dim: (lanes, 1, 1),
                    shared_mem_bytes: lanes * 4,
                },
                &[
                    Arg::P(wide.cached_ptr()),
                    Arg::P(partials.cached_ptr()),
                    Arg::I(rows as i32),
                    Arg::L(cols as i64),
                    Arg::L(col0 as i64),
                    Arg::I(0),
                ],
            );
        }
        assert_bits(
            &format!("axis-0 reduction of {rows} rows"),
            &read(st, narrow.cached_ptr(), cols),
            &read(st, wide.cached_ptr(), cols),
        );
    }
}

/// Past 2^31 elements of the projection the split's `_i64` build writes
/// every row of a window what the `int` build writes that window alone:
/// the last rows, whose offsets pass the range, and the first ones.
#[test]
#[ignore = "8.6 GiB of projection; run on a 32 GB card"]
fn the_split_i64_build_writes_rows_past_i32_max_as_the_int_build_writes_them() {
    let base = Split {
        rows: 0,
        di: 8,
        ng: 1,
        ds: 512,
        nh: 1,
        na: 2,
    };
    let rows = i32::MAX as usize / base.width() + 4096;
    let s = Split { rows, ..base };
    assert!(s.rows * s.width() > i32::MAX as usize);
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let m3k = compile(&ctx, 16);

    // The projection is one word everywhere but in the windows compared,
    // so a row read at a wrapped offset reads other bits than its own.
    let total = s.rows * s.width();
    let proj = zeros(&ctx, total);
    fill(st, proj.cached_ptr(), 0x3e80_0000, total);
    let window = 4096;
    for (r, seed) in [(0, 51u32), (s.rows - window, 52)] {
        let rows = floats(&ctx, &det(window * s.width(), seed, 2.0));
        copy(
            st,
            at(proj.cached_ptr(), r * s.width()),
            rows.cached_ptr(),
            window * s.width(),
        );
    }
    // B and C stay null: the split skips them, and the columns they would
    // take still walk the 64-bit offsets of the projection.
    let (io, _bufs) = split_io(&ctx, s, proj.cached_ptr(), false);
    run_split(&ctx, &m3k.m3_split.i64, s, &io);
    for r in [0, s.rows - window] {
        let one = Split { rows: window, ..s };
        let (own, _own_bufs) = split_io(&ctx, one, at(proj.cached_ptr(), r * s.width()), false);
        let own = SplitIo {
            bias: io.bias,
            ..own
        };
        run_split(&ctx, &m3k.m3_split.i32, one, &own);
        for ((name, a, n), (_, b, _)) in io
            .outputs(s, r, window)
            .into_iter()
            .zip(own.outputs(one, 0, window))
        {
            assert_bits(
                &format!("split rows from {r}: {name}"),
                &read(st, a, n),
                &read(st, b, n),
            );
        }
    }
}

/// Past 2^31 columns the wide axis-0 reduction writes every column what the
/// plain reduction writes it, in launches of at most the grid's x extent.
#[test]
#[ignore = "17 GiB of partials and sums; run on a 32 GB card"]
fn the_wide_axis0_reduction_sums_columns_past_i32_max() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let m3k = compile(&ctx, 16);
    let cols = i32::MAX as usize + (1 << 20);
    let partials = zeros(&ctx, cols);
    let out = zeros(&ctx, cols);
    fill(st, partials.cached_ptr(), 0x3e80_0000, cols);
    let window = 1 << 16;
    let windows = [0, i32::MAX as usize - window / 2, cols - window];
    for (i, &c) in windows.iter().enumerate() {
        let vals = floats(&ctx, &det(window, 60 + i as u32, 1.0));
        copy(st, at(partials.cached_ptr(), c), vals.cached_ptr(), window);
    }
    let lanes = 32u32;
    let mut col0 = 0usize;
    while col0 < cols {
        let n = (cols - col0).min(i32::MAX as usize);
        launch(
            &ctx,
            &m3k.reduce_sum_axis0_wide,
            LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (lanes, 1, 1),
                shared_mem_bytes: lanes * 4,
            },
            &[
                Arg::P(out.cached_ptr()),
                Arg::P(partials.cached_ptr()),
                Arg::I(1),
                Arg::L(cols as i64),
                Arg::L(col0 as i64),
                Arg::I(0),
            ],
        );
        col0 += n;
    }
    for &c in &windows {
        let own = zeros(&ctx, window);
        launch(
            &ctx,
            &m3k.reduce_sum_axis0,
            LaunchConfig {
                grid_dim: (window as u32, 1, 1),
                block_dim: (lanes, 1, 1),
                shared_mem_bytes: lanes * 4,
            },
            &[
                Arg::P(own.cached_ptr()),
                Arg::P(at(partials.cached_ptr(), c)),
                Arg::I(1),
                Arg::I(window as i32),
                Arg::I(0),
            ],
        );
        assert_bits(
            &format!("columns from {c}"),
            &read(st, at(out.cached_ptr(), c), window),
            &read(st, own.cached_ptr(), window),
        );
    }
}

/// Past 2^31 elements of the sequential tape the `_i64` builds of the
/// forward and backward give the last sequence of a batch, whose tape lies
/// wholly past the range, exactly what that sequence gives alone on the
/// `int` builds.
#[test]
#[ignore = "13 GiB of tape; run on a 32 GB card"]
fn the_sequential_i64_builds_give_a_sequence_past_i32_max_its_own_bits() {
    let s = Seq {
        b: 3,
        t: 524_289,
        nh: 1,
        hd: 32,
        ds: 64,
    };
    let one = Seq { b: 1, ..s };
    assert!((s.b - 1) * s.tape() > i32::MAX as usize);
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let m3k = compile(&ctx, 64);
    let bufs = seq_bufs(&ctx, s);
    let all = seq_ptrs(&bufs, s, 0);
    run_seq(
        &ctx,
        &m3k.m3_burnin_fwd.i64,
        &m3k.m3_backward_seq.i64,
        s,
        &all,
    );
    let last = s.b - 1;
    let own = seq_bufs(&ctx, one);
    // The last sequence's inputs, copied out; its starting states, which the
    // batched run overwrote, regenerated from their seeds.
    let mut starts = Vec::new();
    for (i, &(_, per, seed)) in SEQ_BUFS.iter().enumerate() {
        let Some(seed) = seed else { continue };
        let n = per.len(s);
        match per {
            Per::Shared => {}
            Per::State | Per::KState | Per::VState => {
                let all = det(s.b * n, seed, 0.5);
                let start = floats(&ctx, &all[last * n..]);
                copy(st, own[i].cached_ptr(), start.cached_ptr(), n);
                starts.push(start);
            }
            _ => copy(
                st,
                own[i].cached_ptr(),
                at(bufs[i].cached_ptr(), last * n),
                n,
            ),
        }
    }
    run_seq(
        &ctx,
        &m3k.m3_burnin_fwd.i32,
        &m3k.m3_backward_seq.i32,
        one,
        &seq_ptrs(&own, one, 0),
    );
    for (i, &(name, per, _)) in SEQ_BUFS.iter().enumerate() {
        if matches!(per, Per::Shared) {
            continue;
        }
        let n = per.len(s);
        assert_bits(
            &format!("sequence {last} {name}"),
            &read(st, at(bufs[i].cached_ptr(), last * n), n),
            &read(st, own[i].cached_ptr(), n),
        );
    }
}

/// Past 2^31 elements of Q and K the chunk scan's `_i64` build gives the
/// last chunks of a window exactly what the `int` build gives them alone.
#[test]
#[ignore = "20 GiB of Q, K and states; run on a 32 GB card"]
fn the_chunk_scan_i64_build_gives_chunks_past_i32_max_their_own_bits() {
    let s = Chunked {
        b: 1,
        t: 8_392_704,
        nh: 8,
        hd: 8,
        ds: 32,
    };
    assert!(s.b * s.t * s.nh * s.ds > i32::MAX as usize);
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let m3k = compile(&ctx, 32);
    let (io, bufs) = chunk_bufs(&ctx, s);
    let (arm, cfg) = chunk_scan_cfg(s.b, s.nc(), s.nh, s.hd, s.ds, CS);
    let [narrow, wide] = scan_builds(&m3k, arm);
    launch_runs(&ctx, wide, cfg, &io.args(s));
    let chunks = 64;
    let window = Chunked {
        t: chunks * CS,
        ..s
    };
    let di = s.nh * s.hd;
    for c in [0, s.nc() - chunks] {
        let own_y = zeros(&ctx, window.t * di);
        let own = ChunkIo {
            y: own_y.cached_ptr(),
            ..io.past_chunks(s, c)
        };
        let (own_arm, own_cfg) =
            chunk_scan_cfg(window.b, window.nc(), window.nh, window.hd, window.ds, CS);
        assert_eq!(own_arm, arm, "the window takes the whole scan's arm");
        launch_runs(&ctx, narrow, own_cfg, &own.args(window));
        assert_bits(
            &format!("chunks from {c}"),
            &read(st, at(io.y, c * CS * di), window.t * di),
            &read(st, own_y.cached_ptr(), window.t * di),
        );
    }
    drop(bufs);
}
