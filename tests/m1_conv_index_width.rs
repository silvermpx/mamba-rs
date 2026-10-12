//! The two builds of the Mamba-1 conv kernels. Each is compiled with `int`
//! offsets and with `long long` offsets (the `_i64` entry), and a launch
//! takes the `int` build while every offset it forms fits the 32-bit range.
//! The decode step's two builds give the same bits; the other kernels'
//! builds are compared in `m1_conv_any_bits`. Past 2^31 the `_i64` builds
//! give every sequence of a batch exactly what that sequence gives alone.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::sys::CUresult::CUDA_SUCCESS;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{
    IndexWidth, conv_index_width, grid_1d_cover, grid_conv_fwd_tiled, grid_conv_tiled,
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

/// A kernel argument: a device pointer or an int.
#[derive(Clone)]
enum Arg {
    P(u64),
    I(i32),
}

fn launch(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    let mut bld = ctx.stream.launch_builder(f);
    for a in args {
        match a {
            Arg::P(v) => {
                bld.arg(v);
            }
            Arg::I(v) => {
                bld.arg(v);
            }
        }
    }
    unsafe { bld.launch(cfg) }.expect("conv launch");
}

/// The decode step's conv in both offset widths: the four-tap register
/// path and the generic window, x read through the in_proj row stride.
#[test]
fn both_builds_of_the_conv_step_give_the_same_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let (b, di) = (5usize, 40usize);
    let xs = 2 * di;
    for dc in [3usize, 4] {
        for dt in DTYPES {
            let run = |width: IndexWidth| -> Vec<Vec<u8>> {
                let out = DtypedBuf::zeros(&ctx.stream, b * di, dt).expect("out");
                let state = floats(&ctx, &det(b * di * dc, 1, 0.5));
                let x = typed(&ctx, &det(b * xs, 2, 0.7), dt);
                let w = floats(&ctx, &det(di * dc, 3, 0.4));
                let bias = floats(&ctx, &det(di, 4, 0.1));
                launch(
                    &ctx,
                    ctx.kernels.conv1d_step_fwd_silu_typed.get(width).get(dt),
                    grid_1d_cover(b * di).expect("grid"),
                    &[
                        Arg::P(out.cached_ptr()),
                        Arg::P(state.cached_ptr()),
                        Arg::P(x.cached_ptr()),
                        Arg::I(xs as i32),
                        Arg::P(w.cached_ptr()),
                        Arg::P(bias.cached_ptr()),
                        Arg::I(b as i32),
                        Arg::I(di as i32),
                        Arg::I(dc as i32),
                    ],
                );
                vec![
                    read(&ctx.stream, out.cached_ptr(), out.size_bytes()),
                    read(&ctx.stream, state.cached_ptr(), state.size_bytes()),
                ]
            };
            assert!(
                run(IndexWidth::I32) == run(IndexWidth::I64),
                "{dt:?} d_conv {dc}: the conv step builds differ"
            );
        }
    }
}

#[derive(Clone, Copy)]
struct Shape {
    b: usize,
    t: usize,
    di: usize,
    dc: usize,
    /// The x rows' stride, the in_proj output's 2 * d_inner.
    xs: usize,
}

impl Shape {
    fn ints(self) -> [Arg; 4] {
        [
            Arg::I(self.b as i32),
            Arg::I(self.t as i32),
            Arg::I(self.di as i32),
            Arg::I(self.dc as i32),
        ]
    }
}

/// 1025 sequences of 1024 steps over 1024 channels, x read through rows of
/// 2048 columns: the x rows end past 2^31 elements, so every conv launch
/// takes its `_i64` build. The first and the last sequence come out exactly
/// as each one does in a launch of its own on the int build: the training
/// forward's outputs, carry-out and saved window, the prefill forward's
/// t-major copy, and the backward's x gradient and tap and bias partials.
#[test]
#[ignore = "allocates 10 GiB of device memory"]
fn the_conv_i64_builds_give_each_sequence_of_a_batch_past_i32_max_its_own_bits() {
    let s = Shape {
        b: 1025,
        t: 1024,
        di: 1024,
        dc: 4,
        xs: 2048,
    };
    let one = Shape { b: 1, ..s };
    assert_eq!(
        conv_index_width(s.b, s.t, s.di, s.dc, s.xs),
        IndexWidth::I64
    );
    assert_eq!(conv_index_width(1, s.t, s.di, s.dc, s.xs), IndexWidth::I32);

    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let dt = WeightDtype::Bf16;
    let es = dt.size_bytes();
    let k = &ctx.kernels;
    let seqs = [0, s.b - 1];
    let (rows, win) = (s.t * s.xs, s.di * s.dc);
    let n_tiles = s.t.div_ceil(128);

    // Every x row holds 0.25 except the rows of the two checked sequences,
    // which are their own: a wrapped offset would read 0.25 or fault.
    let x = DtypedBuf::zeros(st, s.b * rows, dt).expect("4 GiB of bf16");
    let r = unsafe {
        cudarc::driver::sys::cuMemsetD16Async(x.cached_ptr(), 0x3e80, s.b * rows, st.cu_stream())
    };
    assert_eq!(r, CUDA_SUCCESS, "memset");
    let x_own: Vec<DtypedBuf> = seqs
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            let own = typed(&ctx, &det(rows, 50 + i as u32, 0.6), dt);
            copy(
                st,
                x.cached_ptr() + (q * rows * es) as u64,
                own.cached_ptr(),
                rows * es,
            );
            own
        })
        .collect();
    let w = floats(&ctx, &det(win, 60, 0.4));
    let bias = floats(&ctx, &det(s.di, 61, 0.1));
    let state0 = floats(&ctx, &det(s.b * win, 62, 0.3));
    // A copy of the carry-in state, the batch's or one sequence's.
    let state = |seq: Option<usize>| -> GpuBuffer {
        let (at, n) = seq.map_or((0, s.b * win), |q| (q * win, win));
        let h = GpuBuffer::zeros(st, n).expect("state");
        copy(
            st,
            h.cached_ptr(),
            state0.cached_ptr() + (at * 4) as u64,
            n * 4,
        );
        h
    };
    let act = s.t * s.di;

    // The training forward: outputs, carry-out, saved window.
    {
        let u = DtypedBuf::zeros(st, s.b * act, dt).expect("u");
        let h = state(None);
        let saved = GpuBuffer::zeros(st, s.b * win).expect("saved");
        let mut args = vec![
            Arg::P(u.cached_ptr()),
            Arg::P(h.cached_ptr()),
            Arg::P(saved.cached_ptr()),
            Arg::P(x.cached_ptr()),
            Arg::P(w.cached_ptr()),
            Arg::P(bias.cached_ptr()),
        ];
        args.extend(s.ints());
        args.push(Arg::I(s.xs as i32));
        launch(
            &ctx,
            k.conv1d_burnin_fwd_tiled_typed.i64.get(dt),
            grid_conv_fwd_tiled(s.b, s.di, s.t).expect("grid"),
            &args,
        );
        for (&q, own) in seqs.iter().zip(&x_own) {
            let u1 = DtypedBuf::zeros(st, act, dt).expect("u");
            let h1 = state(Some(q));
            let saved1 = GpuBuffer::zeros(st, win).expect("saved");
            let mut args = vec![
                Arg::P(u1.cached_ptr()),
                Arg::P(h1.cached_ptr()),
                Arg::P(saved1.cached_ptr()),
                Arg::P(own.cached_ptr()),
                Arg::P(w.cached_ptr()),
                Arg::P(bias.cached_ptr()),
            ];
            args.extend(one.ints());
            args.push(Arg::I(s.xs as i32));
            launch(
                &ctx,
                k.conv1d_burnin_fwd_tiled_typed.i32.get(dt),
                grid_conv_fwd_tiled(1, s.di, s.t).expect("grid"),
                &args,
            );
            let pairs = [
                (
                    "outputs",
                    u.cached_ptr() + (q * act * es) as u64,
                    u1.cached_ptr(),
                    act * es,
                ),
                (
                    "carry-out",
                    h.cached_ptr() + (q * win * 4) as u64,
                    h1.cached_ptr(),
                    win * 4,
                ),
                (
                    "saved window",
                    saved.cached_ptr() + (q * win * 4) as u64,
                    saved1.cached_ptr(),
                    win * 4,
                ),
            ];
            for (what, batch, own, bytes) in pairs {
                assert!(
                    read(st, batch, bytes) == read(st, own, bytes),
                    "training forward: {what} of sequence {q} differs from its own launch"
                );
            }
        }
    }

    // The prefill forward with its t-major copy.
    {
        let uts = s.t;
        let u = DtypedBuf::zeros(st, s.b * act, dt).expect("u");
        let u_t = DtypedBuf::zeros(st, s.b * s.di * uts, dt).expect("u_t");
        let h = state(None);
        let mut args = vec![
            Arg::P(u.cached_ptr()),
            Arg::P(u_t.cached_ptr()),
            Arg::P(h.cached_ptr()),
            Arg::P(x.cached_ptr()),
            Arg::P(w.cached_ptr()),
            Arg::P(bias.cached_ptr()),
        ];
        args.extend(s.ints());
        args.push(Arg::I(s.xs as i32));
        args.push(Arg::I(uts as i32));
        launch(
            &ctx,
            k.conv1d_burnin_nosave_tiled_typed.i64.get(dt),
            grid_conv_fwd_tiled(s.b, s.di, s.t).expect("grid"),
            &args,
        );
        for (&q, own) in seqs.iter().zip(&x_own) {
            let u1 = DtypedBuf::zeros(st, act, dt).expect("u");
            let u_t1 = DtypedBuf::zeros(st, s.di * uts, dt).expect("u_t");
            let h1 = state(Some(q));
            let mut args = vec![
                Arg::P(u1.cached_ptr()),
                Arg::P(u_t1.cached_ptr()),
                Arg::P(h1.cached_ptr()),
                Arg::P(own.cached_ptr()),
                Arg::P(w.cached_ptr()),
                Arg::P(bias.cached_ptr()),
            ];
            args.extend(one.ints());
            args.push(Arg::I(s.xs as i32));
            args.push(Arg::I(uts as i32));
            launch(
                &ctx,
                k.conv1d_burnin_nosave_tiled_typed.i32.get(dt),
                grid_conv_fwd_tiled(1, s.di, s.t).expect("grid"),
                &args,
            );
            let t_rows = s.di * uts;
            let pairs = [
                (
                    "outputs",
                    u.cached_ptr() + (q * act * es) as u64,
                    u1.cached_ptr(),
                    act * es,
                ),
                (
                    "t-major outputs",
                    u_t.cached_ptr() + (q * t_rows * es) as u64,
                    u_t1.cached_ptr(),
                    t_rows * es,
                ),
                (
                    "carry-out",
                    h.cached_ptr() + (q * win * 4) as u64,
                    h1.cached_ptr(),
                    win * 4,
                ),
            ];
            for (what, batch, own, bytes) in pairs {
                assert!(
                    read(st, batch, bytes) == read(st, own, bytes),
                    "prefill forward: {what} of sequence {q} differs from its own launch"
                );
            }
        }
    }

    // The backward into the x half of [b * t, 2 * di] gradient rows.
    {
        let d_u = DtypedBuf::zeros(st, s.b * act, dt).expect("d_u");
        let r = unsafe {
            cudarc::driver::sys::cuMemsetD16Async(
                d_u.cached_ptr(),
                0x3d80,
                s.b * act,
                st.cu_stream(),
            )
        };
        assert_eq!(r, CUDA_SUCCESS, "memset");
        let d_u_own: Vec<DtypedBuf> = seqs
            .iter()
            .enumerate()
            .map(|(i, &q)| {
                let own = typed(&ctx, &det(act, 70 + i as u32, 0.2), dt);
                copy(
                    st,
                    d_u.cached_ptr() + (q * act * es) as u64,
                    own.cached_ptr(),
                    act * es,
                );
                own
            })
            .collect();
        let d_x = DtypedBuf::zeros(st, s.b * rows, dt).expect("d_x");
        let (wp_n, bp_n) = (n_tiles * win, n_tiles * s.di);
        let wp = GpuBuffer::zeros(st, s.b * wp_n).expect("tap partials");
        let bp = GpuBuffer::zeros(st, s.b * bp_n).expect("bias partials");
        let tail = |sh: Shape| -> Vec<Arg> {
            let mut t = sh.ints().to_vec();
            t.extend([Arg::I(s.xs as i32), Arg::I(0), Arg::I(s.xs as i32)]);
            t
        };
        let mut args = vec![
            Arg::P(d_x.cached_ptr()),
            Arg::P(wp.cached_ptr()),
            Arg::P(bp.cached_ptr()),
            Arg::P(d_u.cached_ptr()),
            Arg::P(x.cached_ptr()),
            Arg::P(state0.cached_ptr()),
            Arg::P(w.cached_ptr()),
            Arg::P(bias.cached_ptr()),
        ];
        args.extend(tail(s));
        launch(
            &ctx,
            k.conv1d_bwd_tiled_typed.i64.get(dt),
            grid_conv_tiled(s.b, s.di, s.t).expect("grid"),
            &args,
        );
        for ((&q, own), du_own) in seqs.iter().zip(&x_own).zip(&d_u_own) {
            let d_x1 = DtypedBuf::zeros(st, rows, dt).expect("d_x");
            let wp1 = GpuBuffer::zeros(st, wp_n).expect("tap partials");
            let bp1 = GpuBuffer::zeros(st, bp_n).expect("bias partials");
            let init1 = state(Some(q));
            let mut args = vec![
                Arg::P(d_x1.cached_ptr()),
                Arg::P(wp1.cached_ptr()),
                Arg::P(bp1.cached_ptr()),
                Arg::P(du_own.cached_ptr()),
                Arg::P(own.cached_ptr()),
                Arg::P(init1.cached_ptr()),
                Arg::P(w.cached_ptr()),
                Arg::P(bias.cached_ptr()),
            ];
            args.extend(tail(one));
            launch(
                &ctx,
                k.conv1d_bwd_tiled_typed.i32.get(dt),
                grid_conv_tiled(1, s.di, s.t).expect("grid"),
                &args,
            );
            let pairs = [
                (
                    "x gradient",
                    d_x.cached_ptr() + (q * rows * es) as u64,
                    d_x1.cached_ptr(),
                    rows * es,
                ),
                (
                    "tap partials",
                    wp.cached_ptr() + (q * wp_n * 4) as u64,
                    wp1.cached_ptr(),
                    wp_n * 4,
                ),
                (
                    "bias partials",
                    bp.cached_ptr() + (q * bp_n * 4) as u64,
                    bp1.cached_ptr(),
                    bp_n * 4,
                ),
            ];
            for (what, batch, own, bytes) in pairs {
                assert!(
                    read(st, batch, bytes) == read(st, own, bytes),
                    "backward: {what} of sequence {q} differs from its own launch"
                );
            }
        }
    }
}
