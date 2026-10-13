//! The two builds of the Mamba-1 parallel scan forwards. Each forward body is
//! compiled with `int` offsets and with `long long` offsets (the `_i64`
//! entry), and a launch takes the `int` build while every offset it forms
//! fits the 32-bit range. So the two builds must give the same bits wherever
//! both can run, and past 2^31 the `_i64` build must give every sequence of a
//! batch exactly what that sequence gives alone.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::sys::CUresult::CUDA_SUCCESS;
use cudarc::driver::{CudaFunction, CudaStream, PushKernelArg};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{
    IndexWidth, grid_parallel_scan, scan_fwd_index_width, scan_fwd_nosave_index_width,
    scan_tape_len, y_runs,
};

const DTYPES: [WeightDtype; 3] = [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16];

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

/// `bytes` bytes at `ptr`, read once the stream drains.
fn read(stream: &Arc<CudaStream>, ptr: u64, bytes: usize) -> Vec<u8> {
    stream.synchronize().expect("sync");
    let mut out = vec![0u8; bytes];
    let r = unsafe { cudarc::driver::sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), ptr, bytes) };
    assert_eq!(r, CUDA_SUCCESS, "read");
    out
}

/// Copy `bytes` bytes from `src` to `dst` on the stream.
fn copy(stream: &Arc<CudaStream>, dst: u64, src: u64, bytes: usize) {
    let r =
        unsafe { cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.cu_stream()) };
    assert_eq!(r, CUDA_SUCCESS, "copy");
}

fn typed(ctx: &GpuCtx, data: &[f32], dt: WeightDtype) -> DtypedBuf {
    let buf = DtypedBuf::zeros(&ctx.stream, data.len(), dt).expect("alloc");
    ctx.stream.synchronize().expect("sync");
    buf.upload_f32(&ctx.stream, data).expect("upload");
    buf
}

fn floats(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut buf = GpuBuffer::zeros(&ctx.stream, data.len()).expect("alloc");
    ctx.stream.synchronize().expect("sync");
    buf.upload(&ctx.stream, data).expect("upload");
    buf
}

/// A negative A row per channel and state, as the model keeps `-exp(a_log)`.
fn a_neg(n: usize, seed: u32) -> Vec<f32> {
    det(n, seed, 0.6)
        .iter()
        .map(|x| -(x.abs() + 0.05))
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    b: usize,
    t: usize,
    di: usize,
    ds: usize,
}

impl Shape {
    fn args(self) -> [i32; 4] {
        [self.b as i32, self.t as i32, self.di as i32, self.ds as i32]
    }

    /// Floats of the h tape one sequence writes.
    fn tape_per_seq(self, slim: bool) -> usize {
        if slim {
            scan_tape_len(1, self.t, self.di, self.ds)
        } else {
            (self.t + 1) * self.di * self.ds
        }
    }
}

/// Device pointers of one saving forward. The tape goes in as both
/// `h_saved` and `run_tape`, as the model passes it.
struct SaveIo {
    h: u64,
    y: u64,
    tape: u64,
    delta_raw: u64,
    delta_saved: u64,
    u: u64,
    bm: u64,
    cm: u64,
    a_neg: u64,
    d: u64,
}

fn launch_save(ctx: &GpuCtx, f: &CudaFunction, io: &SaveIo, s: Shape, slim: bool) {
    let [b, t, di, ds] = s.args();
    let slim = i32::from(slim);
    for (y_first, cfg) in y_runs(grid_parallel_scan(s.b, s.di, s.ds)) {
        let mut bld = ctx.stream.launch_builder(f);
        bld.arg(&io.h);
        bld.arg(&io.y);
        bld.arg(&io.tape);
        bld.arg(&io.delta_raw);
        bld.arg(&io.delta_saved);
        bld.arg(&io.u);
        bld.arg(&io.bm);
        bld.arg(&io.cm);
        bld.arg(&io.a_neg);
        bld.arg(&io.d);
        bld.arg(&b);
        bld.arg(&t);
        bld.arg(&di);
        bld.arg(&ds);
        bld.arg(&io.tape);
        bld.arg(&slim);
        bld.arg(&y_first);
        unsafe { bld.launch(cfg) }.expect("saving forward");
    }
}

/// How the forward without saves lays out its activations and stores y.
#[derive(Clone, Copy, Debug)]
enum Rows {
    /// Row-major `[batch, T, d_inner]`, y stored as is.
    Plain,
    /// Row-major, y multiplied by SiLU of the gate half of the in_proj rows.
    Gated,
    /// T-major `[batch, d_inner, stride]`, y stored ungated.
    TMajor(usize),
}

impl Rows {
    /// Elements of delta, u or y for one sequence.
    fn per_seq(self, s: Shape) -> usize {
        match self {
            Rows::TMajor(stride) => s.di * stride,
            _ => s.t * s.di,
        }
    }
}

/// Device pointers of one forward without saves.
struct NosaveIo {
    h: u64,
    y: u64,
    delta: u64,
    u: u64,
    bm: u64,
    cm: u64,
    a_neg: u64,
    d: u64,
    gate: u64,
}

fn launch_nosave(ctx: &GpuCtx, f: &CudaFunction, io: &NosaveIo, s: Shape, rows: Rows) {
    let [b, t, di, ds] = s.args();
    // The model passes the in_proj stride on every route; only the row-major
    // gated store reads it.
    let gate_stride = match rows {
        Rows::Plain => 0i32,
        _ => 2 * di,
    };
    let tstride = match rows {
        Rows::TMajor(stride) => stride as i32,
        _ => 0i32,
    };
    for (y_first, cfg) in y_runs(grid_parallel_scan(s.b, s.di, s.ds)) {
        let mut bld = ctx.stream.launch_builder(f);
        bld.arg(&io.h);
        bld.arg(&io.y);
        bld.arg(&io.delta);
        bld.arg(&io.u);
        bld.arg(&io.bm);
        bld.arg(&io.cm);
        bld.arg(&io.a_neg);
        bld.arg(&io.d);
        bld.arg(&io.gate);
        bld.arg(&gate_stride);
        bld.arg(&b);
        bld.arg(&t);
        bld.arg(&di);
        bld.arg(&ds);
        bld.arg(&tstride);
        bld.arg(&y_first);
        unsafe { bld.launch(cfg) }.expect("forward without saves");
    }
}

/// Every output of one saving forward on fixed inputs, as bytes.
fn run_save(ctx: &GpuCtx, f: &CudaFunction, s: Shape, dt: WeightDtype, slim: bool) -> Vec<Vec<u8>> {
    let (bt, st) = (s.b * s.t, s.b * s.di * s.ds);
    let h = floats(ctx, &det(st, 1, 0.8));
    let y = DtypedBuf::zeros(&ctx.stream, bt * s.di, dt).expect("y");
    let tape = GpuBuffer::zeros(&ctx.stream, s.b * s.tape_per_seq(slim)).expect("tape");
    let delta_raw = typed(ctx, &det(bt * s.di, 2, 3.0), dt);
    let delta_saved = DtypedBuf::zeros(&ctx.stream, bt * s.di, dt).expect("delta");
    let u = typed(ctx, &det(bt * s.di, 3, 1.0), dt);
    let bm = typed(ctx, &det(bt * s.ds, 4, 0.7), dt);
    let cm = typed(ctx, &det(bt * s.ds, 5, 0.7), dt);
    let an = floats(ctx, &a_neg(s.di * s.ds, 6));
    let d = floats(ctx, &det(s.di, 7, 1.0));
    let io = SaveIo {
        h: h.cached_ptr(),
        y: y.cached_ptr(),
        tape: tape.cached_ptr(),
        delta_raw: delta_raw.cached_ptr(),
        delta_saved: delta_saved.cached_ptr(),
        u: u.cached_ptr(),
        bm: bm.cached_ptr(),
        cm: cm.cached_ptr(),
        a_neg: an.cached_ptr(),
        d: d.cached_ptr(),
    };
    launch_save(ctx, f, &io, s, slim);
    vec![
        read(&ctx.stream, io.h, h.size_bytes()),
        read(&ctx.stream, io.y, y.size_bytes()),
        read(&ctx.stream, io.delta_saved, delta_saved.size_bytes()),
        read(&ctx.stream, io.tape, tape.size_bytes()),
    ]
}

/// Every output of one forward without saves on fixed inputs, as bytes.
fn run_nosave(
    ctx: &GpuCtx,
    f: &CudaFunction,
    s: Shape,
    dt: WeightDtype,
    rows: Rows,
) -> Vec<Vec<u8>> {
    let (bt, st, act) = (s.b * s.t, s.b * s.di * s.ds, s.b * rows.per_seq(s));
    let h = floats(ctx, &det(st, 11, 0.8));
    let y = DtypedBuf::zeros(&ctx.stream, act, dt).expect("y");
    let delta: Vec<f32> = det(act, 12, 0.3).iter().map(|x| x.abs() + 0.01).collect();
    let delta = typed(ctx, &delta, dt);
    let u = typed(ctx, &det(act, 13, 1.0), dt);
    let bm = typed(ctx, &det(bt * s.ds, 14, 0.7), dt);
    let cm = typed(ctx, &det(bt * s.ds, 15, 0.7), dt);
    let an = floats(ctx, &a_neg(s.di * s.ds, 16));
    let d = floats(ctx, &det(s.di, 17, 1.0));
    let gate = typed(ctx, &det(bt * 2 * s.di, 18, 2.0), dt);
    let io = NosaveIo {
        h: h.cached_ptr(),
        y: y.cached_ptr(),
        delta: delta.cached_ptr(),
        u: u.cached_ptr(),
        bm: bm.cached_ptr(),
        cm: cm.cached_ptr(),
        a_neg: an.cached_ptr(),
        d: d.cached_ptr(),
        gate: gate.cached_ptr(),
    };
    launch_nosave(ctx, f, &io, s, rows);
    vec![
        read(&ctx.stream, io.h, h.size_bytes()),
        read(&ctx.stream, io.y, y.size_bytes()),
    ]
}

/// Shapes with a partial last chunk, several chunks, a state count that is
/// not a power of two and a channel count that is not a multiple of four.
const SHAPES: [Shape; 2] = [
    Shape {
        b: 3,
        t: 2500,
        di: 40,
        ds: 16,
    },
    Shape {
        b: 2,
        t: 777,
        di: 22,
        ds: 7,
    },
];

#[test]
fn both_builds_of_the_saving_forward_give_the_same_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let k = &ctx.kernels.ssm_parallel_fwd_typed;
    for s in SHAPES {
        for dt in DTYPES {
            for slim in [true, false] {
                let narrow = run_save(&ctx, k.i32.get(dt), s, dt, slim);
                let wide = run_save(&ctx, k.i64.get(dt), s, dt, slim);
                for (i, what) in ["h", "y", "delta", "tape"].iter().enumerate() {
                    assert!(
                        narrow[i] == wide[i],
                        "{what} differs between the builds: {s:?} {dt:?} slim {slim}"
                    );
                }
            }
        }
    }
}

#[test]
fn both_builds_of_the_forward_without_saves_give_the_same_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let k = &ctx.kernels.ssm_parallel_fwd_nosave_typed;
    for s in SHAPES {
        for dt in DTYPES {
            for rows in [
                Rows::Plain,
                Rows::Gated,
                Rows::TMajor(s.t.next_multiple_of(8)),
            ] {
                let narrow = run_nosave(&ctx, k.i32.get(dt), s, dt, rows);
                let wide = run_nosave(&ctx, k.i64.get(dt), s, dt, rows);
                for (i, what) in ["h", "y"].iter().enumerate() {
                    assert!(
                        narrow[i] == wide[i],
                        "{what} differs between the builds: {s:?} {dt:?} {rows:?}"
                    );
                }
            }
        }
    }
}

/// One sequence of a batch, carried into a launch of its own.
struct Alone {
    seq: usize,
    /// The sequence's B and C rows, `[d_state, T]` each.
    bm: DtypedBuf,
    cm: DtypedBuf,
}

/// 8193 sequences of 1024 steps over 256 states: the B and C rows of the
/// last sequence start at element 2^31, so the int build would wrap every
/// offset into them. The launch takes the `_i64` build, and the first and
/// the last sequence come out exactly as each one does in a launch of its
/// own on the int build: y, the final state, the post-softplus save and the
/// tape of both kinds, and y of every row layout without saves.
#[test]
#[ignore = "allocates 16 GiB of device memory"]
fn the_i64_builds_give_each_sequence_of_a_batch_past_i32_max_its_own_bits() {
    let s = Shape {
        b: 8193,
        t: 1024,
        di: 1,
        ds: 256,
    };
    let one = Shape { b: 1, ..s };
    assert_eq!(
        scan_fwd_index_width(s.b, s.t, s.di, s.ds, true),
        IndexWidth::I64
    );
    assert_eq!(
        scan_fwd_nosave_index_width(s.b, s.t, s.di, s.ds, 2 * s.di, false),
        IndexWidth::I64
    );
    assert_eq!(
        scan_fwd_index_width(1, s.t, s.di, s.ds, false),
        IndexWidth::I32
    );

    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let dt = WeightDtype::Bf16;
    let es = dt.size_bytes();
    let (bt, slab) = (s.b * s.t, s.ds * s.t);

    // Every row of B and C holds 0.25 except the two checked sequences,
    // whose rows are their own: a wrapped offset would read 0.25 rows or
    // fault.
    let big_b = DtypedBuf::zeros(st, s.b * slab, dt).expect("4 GiB of bf16");
    let big_c = DtypedBuf::zeros(st, s.b * slab, dt).expect("4 GiB of bf16");
    for buf in [&big_b, &big_c] {
        let r = unsafe {
            cudarc::driver::sys::cuMemsetD16Async(
                buf.cached_ptr(),
                0x3e80,
                s.b * slab,
                st.cu_stream(),
            )
        };
        assert_eq!(r, CUDA_SUCCESS, "memset");
    }
    let alone: Vec<Alone> = [0, s.b - 1]
        .iter()
        .enumerate()
        .map(|(i, &seq)| {
            let seed = 100 + 2 * i as u32;
            let a = Alone {
                seq,
                bm: typed(&ctx, &det(slab, seed, 0.7), dt),
                cm: typed(&ctx, &det(slab, seed + 1, 0.7), dt),
            };
            let at = (seq * slab * es) as u64;
            copy(st, big_b.cached_ptr() + at, a.bm.cached_ptr(), slab * es);
            copy(st, big_c.cached_ptr() + at, a.cm.cached_ptr(), slab * es);
            a
        })
        .collect();
    let an = floats(&ctx, &a_neg(s.di * s.ds, 30));
    let d = floats(&ctx, &det(s.di, 31, 1.0));
    let h0 = floats(&ctx, &det(s.b * s.di * s.ds, 32, 0.8));
    let state_bytes = s.di * s.ds * 4;

    // A copy of the initial state, the batch's or one sequence's.
    let state = |seq: Option<usize>| -> GpuBuffer {
        let (at, n) = match seq {
            Some(q) => (q * state_bytes, s.di * s.ds),
            None => (0, s.b * s.di * s.ds),
        };
        let h = GpuBuffer::zeros(st, n).expect("state");
        copy(st, h.cached_ptr(), h0.cached_ptr() + at as u64, n * 4);
        h
    };
    // One sequence's rows of a batch-wide activation, `per` elements each.
    let rows_of = |src: &DtypedBuf, seq: usize, per: usize| -> DtypedBuf {
        let dst = DtypedBuf::zeros(st, per, dt).expect("rows");
        copy(
            st,
            dst.cached_ptr(),
            src.cached_ptr() + (seq * per * es) as u64,
            per * es,
        );
        dst
    };

    // Saving forward, slim and full tape.
    let delta_raw = typed(&ctx, &det(bt * s.di, 33, 3.0), dt);
    let u = typed(&ctx, &det(bt * s.di, 34, 1.0), dt);
    for slim in [true, false] {
        let h = state(None);
        let y = DtypedBuf::zeros(st, bt * s.di, dt).expect("y");
        let saved = DtypedBuf::zeros(st, bt * s.di, dt).expect("delta");
        let tape = GpuBuffer::zeros(st, s.b * s.tape_per_seq(slim)).expect("tape");
        let io = SaveIo {
            h: h.cached_ptr(),
            y: y.cached_ptr(),
            tape: tape.cached_ptr(),
            delta_raw: delta_raw.cached_ptr(),
            delta_saved: saved.cached_ptr(),
            u: u.cached_ptr(),
            bm: big_b.cached_ptr(),
            cm: big_c.cached_ptr(),
            a_neg: an.cached_ptr(),
            d: d.cached_ptr(),
        };
        launch_save(
            &ctx,
            ctx.kernels.ssm_parallel_fwd_typed.i64.get(dt),
            &io,
            s,
            slim,
        );
        for a in &alone {
            let act = s.t * s.di;
            let tape_n = s.tape_per_seq(slim);
            let h1 = state(Some(a.seq));
            let raw1 = rows_of(&delta_raw, a.seq, act);
            let u1 = rows_of(&u, a.seq, act);
            let y1 = DtypedBuf::zeros(st, act, dt).expect("y");
            let saved1 = DtypedBuf::zeros(st, act, dt).expect("delta");
            let tape1 = GpuBuffer::zeros(st, tape_n).expect("tape");
            let io1 = SaveIo {
                h: h1.cached_ptr(),
                y: y1.cached_ptr(),
                tape: tape1.cached_ptr(),
                delta_raw: raw1.cached_ptr(),
                delta_saved: saved1.cached_ptr(),
                u: u1.cached_ptr(),
                bm: a.bm.cached_ptr(),
                cm: a.cm.cached_ptr(),
                a_neg: an.cached_ptr(),
                d: d.cached_ptr(),
            };
            launch_save(
                &ctx,
                ctx.kernels.ssm_parallel_fwd_typed.i32.get(dt),
                &io1,
                one,
                slim,
            );
            let q = a.seq;
            let pairs = [
                ("h", io.h + (q * state_bytes) as u64, io1.h, state_bytes),
                ("y", io.y + (q * act * es) as u64, io1.y, act * es),
                (
                    "delta",
                    io.delta_saved + (q * act * es) as u64,
                    io1.delta_saved,
                    act * es,
                ),
                (
                    "tape",
                    io.tape + (q * tape_n * 4) as u64,
                    io1.tape,
                    tape_n * 4,
                ),
            ];
            for (what, batch, own, bytes) in pairs {
                assert!(
                    read(st, batch, bytes) == read(st, own, bytes),
                    "saving forward, slim {slim}: {what} of sequence {q} differs from its own launch"
                );
            }
        }
    }

    // Forward without saves, every row layout.
    let gate = typed(&ctx, &det(bt * 2 * s.di, 35, 2.0), dt);
    for rows in [Rows::Plain, Rows::Gated, Rows::TMajor(s.t)] {
        let act = rows.per_seq(s);
        let delta: Vec<f32> = det(s.b * act, 36, 0.3)
            .iter()
            .map(|x| x.abs() + 0.01)
            .collect();
        let delta = typed(&ctx, &delta, dt);
        let u = typed(&ctx, &det(s.b * act, 37, 1.0), dt);
        let h = state(None);
        let y = DtypedBuf::zeros(st, s.b * act, dt).expect("y");
        let io = NosaveIo {
            h: h.cached_ptr(),
            y: y.cached_ptr(),
            delta: delta.cached_ptr(),
            u: u.cached_ptr(),
            bm: big_b.cached_ptr(),
            cm: big_c.cached_ptr(),
            a_neg: an.cached_ptr(),
            d: d.cached_ptr(),
            gate: gate.cached_ptr(),
        };
        let kernels = &ctx.kernels.ssm_parallel_fwd_nosave_typed;
        launch_nosave(&ctx, kernels.i64.get(dt), &io, s, rows);
        for a in &alone {
            let q = a.seq;
            let h1 = state(Some(q));
            let delta1 = rows_of(&delta, q, act);
            let u1 = rows_of(&u, q, act);
            let gate1 = rows_of(&gate, q, s.t * 2 * s.di);
            let y1 = DtypedBuf::zeros(st, act, dt).expect("y");
            let io1 = NosaveIo {
                h: h1.cached_ptr(),
                y: y1.cached_ptr(),
                delta: delta1.cached_ptr(),
                u: u1.cached_ptr(),
                bm: a.bm.cached_ptr(),
                cm: a.cm.cached_ptr(),
                a_neg: an.cached_ptr(),
                d: d.cached_ptr(),
                gate: gate1.cached_ptr(),
            };
            launch_nosave(&ctx, kernels.i32.get(dt), &io1, one, rows);
            let pairs = [
                ("h", io.h + (q * state_bytes) as u64, io1.h, state_bytes),
                ("y", io.y + (q * act * es) as u64, io1.y, act * es),
            ];
            for (what, batch, own, bytes) in pairs {
                assert!(
                    read(st, batch, bytes) == read(st, own, bytes),
                    "forward without saves, {rows:?}: {what} of sequence {q} differs from its own launch"
                );
            }
        }
    }
}
