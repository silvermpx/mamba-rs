//! The Mamba-1 parallel scan forwards against their released form. A released
//! forward took one channel a block of 128 threads, eight steps a thread, so a
//! window shorter than the block's 1024-step chunk left all but a few of the
//! block's threads stepping the identity. The forwards now hold a group of
//! channels in a block, each in the fewest threads that cover its window, and
//! every output must equal the released kernel's bit for bit: y, the saved
//! deltas, the state and the slim or full tape. One allowance: the released
//! block total passed through the identity steps, which turn a negative zero
//! positive, so a zero may differ in its sign.
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;
#[path = "common/frozen_kernels.rs"]
mod frozen_kernels;

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{
    SCAN_NTHREADS, grid_parallel_scan, scan_group_width, scan_tape_len, y_runs,
};

const DTYPES: [WeightDtype; 3] = [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16];

/// Windows on both sides of every group width's span, eight steps a thread,
/// and of the block's 1024-step chunk.
const WINDOWS: [usize; 22] = [
    1, 7, 8, 9, 63, 64, 65, 100, 128, 129, 255, 256, 257, 300, 511, 512, 513, 777, 1024, 1025,
    1500, 2049,
];

/// `(d_inner, d_state)`: channel counts a group count does not divide, so a
/// block holds groups past the last channel, and states from one to the
/// largest.
const CHANNELS: [(usize, usize); 4] = [(1, 16), (33, 1), (24, 64), (33, 256)];

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
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

fn typed_values(ctx: &GpuCtx, buf: &DtypedBuf, n: usize) -> Vec<f32> {
    let mut v = vec![0f32; n];
    buf.download_f32(&ctx.stream, &mut v).expect("download");
    v
}

/// The bits of `v`, every zero as +0.
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter()
        .map(|x| if *x == 0.0 { 0 } else { x.to_bits() })
        .collect()
}

/// The released forwards' launch: a channel a block on the y axis, with the
/// block-scan rows, the running prefix at d_state and an exchange slot per
/// warp.
fn released_grid(b: usize, di: usize, ds: usize) -> LaunchConfig {
    let nwarps = SCAN_NTHREADS / 32;
    LaunchConfig {
        grid_dim: (b as u32, di as u32, 1),
        block_dim: (SCAN_NTHREADS as u32, 1, 1),
        shared_mem_bytes: ((4 * nwarps + 2 * ds) * std::mem::size_of::<f32>()) as u32,
    }
}

/// A released kernel's entry for `dt`.
fn released(frozen: &Arc<CudaModule>, stem: &str, dt: WeightDtype) -> CudaFunction {
    let suffix = match dt {
        WeightDtype::F32 => "",
        WeightDtype::Bf16 => "_bf16",
        _ => "_f16",
    };
    frozen
        .load_function(&format!("{stem}{suffix}"))
        .expect("released kernel")
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    b: usize,
    t: usize,
    di: usize,
    ds: usize,
}

/// The inputs both forwards read.
struct Inputs {
    delta: DtypedBuf,
    u: DtypedBuf,
    bm: DtypedBuf,
    cm: DtypedBuf,
    gate: DtypedBuf,
    a_neg: GpuBuffer,
    d: GpuBuffer,
    h0: Vec<f32>,
}

impl Inputs {
    /// `delta` holds pre-softplus rows when `raw` (the saving forward
    /// applies the softplus) and post-softplus rows otherwise; `rows` is one
    /// sequence's span of delta, u and y.
    fn new(ctx: &GpuCtx, s: Shape, dt: WeightDtype, (rows, raw): (usize, bool)) -> Self {
        let n = s.b * rows;
        let delta: Vec<f32> = det(n, 11, 1.5)
            .iter()
            .map(|&v| if raw { v } else { v.abs() * 0.2 + 0.01 })
            .collect();
        let a_neg: Vec<f32> = det(s.di * s.ds, 15, 1.0)
            .iter()
            .map(|v| -(v.abs() + 0.1))
            .collect();
        Self {
            delta: typed(ctx, &delta, dt),
            u: typed(ctx, &det(n, 12, 1.0), dt),
            bm: typed(ctx, &det(s.b * s.ds * s.t, 13, 0.7), dt),
            cm: typed(ctx, &det(s.b * s.ds * s.t, 14, 0.7), dt),
            gate: typed(ctx, &det(s.b * s.t * 2 * s.di, 18, 2.0), dt),
            a_neg: floats(ctx, &a_neg),
            d: floats(ctx, &det(s.di, 16, 1.0)),
            h0: det(s.b * s.di * s.ds, 17, 1.0),
        }
    }
}

/// The saving forward's outputs: y, the saved deltas, the state and the tape.
fn run_save(
    ctx: &GpuCtx,
    (f, grid): (&CudaFunction, LaunchConfig),
    (s, dt, slim): (Shape, WeightDtype, bool),
    io: &Inputs,
) -> [Vec<f32>; 4] {
    let st = &ctx.stream;
    let n = s.b * s.t * s.di;
    let tape_len = if slim {
        scan_tape_len(s.b, s.t, s.di, s.ds)
    } else {
        s.b * s.di * s.ds * (s.t + 1)
    };
    let h = floats(ctx, &io.h0);
    let y = DtypedBuf::zeros(st, n, dt).expect("y");
    let delta_saved = DtypedBuf::zeros(st, n, dt).expect("delta");
    let tape = GpuBuffer::zeros(st, tape_len).expect("tape");
    let ints = [s.b, s.t, s.di, s.ds].map(|v| v as i32);
    let slim_i = i32::from(slim);
    let ptrs = [
        h.cached_ptr(),
        y.cached_ptr(),
        tape.cached_ptr(),
        io.delta.cached_ptr(),
        delta_saved.cached_ptr(),
        io.u.cached_ptr(),
        io.bm.cached_ptr(),
        io.cm.cached_ptr(),
        io.a_neg.cached_ptr(),
        io.d.cached_ptr(),
    ];
    let tape_ptr = tape.cached_ptr();
    for (y_first, cfg) in y_runs(grid) {
        let mut bld = st.launch_builder(f);
        for p in &ptrs {
            bld.arg(p);
        }
        for v in &ints {
            bld.arg(v);
        }
        bld.arg(&tape_ptr);
        bld.arg(&slim_i);
        bld.arg(&y_first);
        unsafe { bld.launch(cfg) }.expect("saving forward");
    }
    st.synchronize().expect("sync");
    [
        typed_values(ctx, &y, n),
        typed_values(ctx, &delta_saved, n),
        h.to_cpu(st).expect("state"),
        tape.to_cpu(st).expect("tape"),
    ]
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
    fn per_seq(self, s: Shape) -> usize {
        match self {
            Rows::TMajor(stride) => s.di * stride,
            _ => s.t * s.di,
        }
    }
}

/// The forward without saves' outputs: y and the state.
fn run_nosave(
    ctx: &GpuCtx,
    (f, grid): (&CudaFunction, LaunchConfig),
    (s, dt, rows): (Shape, WeightDtype, Rows),
    io: &Inputs,
) -> [Vec<f32>; 2] {
    let st = &ctx.stream;
    let n = s.b * rows.per_seq(s);
    let h = floats(ctx, &io.h0);
    let y = DtypedBuf::zeros(st, n, dt).expect("y");
    let gate_stride = match rows {
        Rows::Plain => 0i32,
        _ => 2 * s.di as i32,
    };
    let tstride = match rows {
        Rows::TMajor(stride) => stride as i32,
        _ => 0i32,
    };
    let ints = [s.b, s.t, s.di, s.ds].map(|v| v as i32);
    let ptrs = [
        h.cached_ptr(),
        y.cached_ptr(),
        io.delta.cached_ptr(),
        io.u.cached_ptr(),
        io.bm.cached_ptr(),
        io.cm.cached_ptr(),
        io.a_neg.cached_ptr(),
        io.d.cached_ptr(),
        io.gate.cached_ptr(),
    ];
    for (y_first, cfg) in y_runs(grid) {
        let mut bld = st.launch_builder(f);
        for p in &ptrs {
            bld.arg(p);
        }
        bld.arg(&gate_stride);
        for v in &ints {
            bld.arg(v);
        }
        bld.arg(&tstride);
        bld.arg(&y_first);
        unsafe { bld.launch(cfg) }.expect("forward without saves");
    }
    st.synchronize().expect("sync");
    [typed_values(ctx, &y, n), h.to_cpu(st).expect("state")]
}

#[test]
fn m1_parallel_forwards_keep_the_bits_of_their_released_form() {
    let device = GpuDevice::new(0).expect("device");
    let ctx = GpuCtx::new(&device).expect("context");
    let frozen = frozen_kernels::frozen_module(
        device.context(),
        arch::arch0(),
        None,
        include_str!("fixtures/m1_parallel_fwd_reference.cu"),
    );
    let mut cases = 0;
    for t in WINDOWS {
        // The grid divides the channels by the groups a block holds.
        let groups = SCAN_NTHREADS / scan_group_width(t);
        for (di, ds) in CHANNELS {
            let s = Shape { b: 2, t, di, ds };
            assert_eq!(
                grid_parallel_scan(s.b, t, di, ds).grid_dim.1 as usize,
                di.div_ceil(groups)
            );
            for dt in DTYPES {
                let io = Inputs::new(&ctx, s, dt, (t * di, true));
                for slim in [true, false] {
                    let old = run_save(
                        &ctx,
                        (
                            &released(&frozen, "ssm_parallel_scan_fwd", dt),
                            released_grid(s.b, di, ds),
                        ),
                        (s, dt, slim),
                        &io,
                    );
                    let new = run_save(
                        &ctx,
                        (
                            ctx.kernels.ssm_parallel_fwd_typed.i32.get(dt),
                            grid_parallel_scan(s.b, t, di, ds),
                        ),
                        (s, dt, slim),
                        &io,
                    );
                    for (what, (o, n)) in ["y", "saved delta", "state", "tape"]
                        .iter()
                        .zip(old.iter().zip(&new))
                    {
                        assert_eq!(
                            bits(o),
                            bits(n),
                            "the saving forward's {what} at {s:?} {dt:?}, slim {slim}"
                        );
                    }
                    cases += 1;
                }
                for rows in [Rows::Plain, Rows::Gated, Rows::TMajor(t + 3)] {
                    let io = Inputs::new(&ctx, s, dt, (rows.per_seq(s), false));
                    let old = run_nosave(
                        &ctx,
                        (
                            &released(&frozen, "ssm_parallel_scan_fwd_nosave", dt),
                            released_grid(s.b, di, ds),
                        ),
                        (s, dt, rows),
                        &io,
                    );
                    let new = run_nosave(
                        &ctx,
                        (
                            ctx.kernels.ssm_parallel_fwd_nosave_typed.i32.get(dt),
                            grid_parallel_scan(s.b, t, di, ds),
                        ),
                        (s, dt, rows),
                        &io,
                    );
                    for (what, (o, n)) in ["y", "state"].iter().zip(old.iter().zip(&new)) {
                        assert_eq!(
                            bits(o),
                            bits(n),
                            "the forward without saves' {what} at {s:?} {dt:?}, {rows:?}"
                        );
                    }
                    cases += 1;
                }
            }
        }
    }
    println!("m1 parallel forwards: the released kernels' bits on {cases} cases");
}
