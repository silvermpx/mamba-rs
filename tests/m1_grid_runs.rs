//! Mamba-1 kernels over grid axes past the y and z extent of 65535 blocks.
//! Such an axis is launched in runs, each run passing its first block, and
//! every block past the first run must compute what it would in one launch
//! over the whole axis. The conv tiles, the t-major transposes, the B and C
//! gather and the per-sequence column sum are checked against launches of
//! their own over the same rows; the parallel scans, whose channels share
//! their B and C rows, against channels given the same inputs in the first
//! run.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::sys::CUresult::CUDA_SUCCESS;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{
    GRID_YZ_EXTENT, IndexWidth, SCAN_BWD_DGROUP, conv_index_width, grid_1d_wide,
    grid_conv_fwd_tiled, grid_conv_tiled, grid_gather_bc_tiled, grid_parallel_scan,
    grid_parallel_scan_bwd, grid_parallel_scan_bwd_fold, grid_tmajor_tiles, scan_bwd_index_width,
    scan_fwd_index_width, scan_fwd_nosave_index_width, scan_tape_len, y_runs, yz_runs,
};

const RUN: usize = GRID_YZ_EXTENT as usize;

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

/// Fill `bytes` bytes at `ptr` with 0xff, a NaN in every float type, so a
/// value no launch wrote cannot pass for one.
fn poison(stream: &Arc<CudaStream>, ptr: u64, bytes: usize) {
    let r = unsafe { cudarc::driver::sys::cuMemsetD8Async(ptr, 0xff, bytes, stream.cu_stream()) };
    assert_eq!(r, CUDA_SUCCESS, "poison");
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
#[derive(Clone, Copy)]
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
    unsafe { bld.launch(cfg) }.expect("launch");
}

/// [`launch`] once per run of the grid's y axis, each run's first block
/// appended as the kernel's last argument.
fn launch_runs(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    for (y_first, cfg) in y_runs(cfg) {
        let mut run = args.to_vec();
        run.push(Arg::I(y_first));
        launch(ctx, f, cfg, &run);
    }
}

/// [`launch`] once per run of the grid's y and z axes, each run's first y
/// and z blocks appended.
fn launch_yz_runs(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    for (y_first, z_first, cfg) in yz_runs(cfg) {
        let mut run = args.to_vec();
        run.extend([Arg::I(y_first), Arg::I(z_first)]);
        launch(ctx, f, cfg, &run);
    }
}

/// The carry-in a launch over the steps from `t1` on reads: per channel the
/// `dc` x values before `t1`, oldest first, as the serial walk holds them.
fn carry_in(x: &[f32], xs: usize, di: usize, dc: usize, t1: usize) -> Vec<f32> {
    (0..di * dc)
        .map(|i| x[(t1 - dc + i % dc) * xs + i / dc])
        .collect()
}

/// Steps whose tiles of 32 pass the grid's y extent. The forward conv
/// kernels, the saving and the inference tiled ones up to eight taps and
/// the one of any width past them, launch their tiles in runs, and the
/// steps of the second run come out exactly as a launch over just those
/// steps gives them, its carry-in the x rows before them: the outputs, the
/// t-major copy and the carry-out.
#[test]
fn the_conv_forward_gives_tiles_past_the_y_extent_their_own_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let k = &ctx.kernels;
    let dt = WeightDtype::F32;
    let (di, xs) = (8usize, 16usize);
    let t1 = RUN * 32;
    let t = t1 + 109;
    let tw = t - t1;
    assert!(grid_conv_fwd_tiled(1, di, t).unwrap().grid_dim.1 as usize > RUN);
    let width = conv_index_width(1, t, di, 12, xs);
    assert_eq!(width, IndexWidth::I32);
    let (ts, tws) = (t.next_multiple_of(8), tw.next_multiple_of(8));
    for dc in [4usize, 12] {
        let x_host = det(t * xs, 70 + dc as u32, 0.6);
        let x = floats(&ctx, &x_host);
        let w = floats(&ctx, &det(di * dc, 71, 0.4));
        let bias = floats(&ctx, &det(di, 72, 0.1));
        let state0 = det(di * dc, 73, 0.3);
        let window_state = carry_in(&x_host, xs, di, dc, t1);
        // (outputs, t-major copy, state) of one launch from `state` over
        // `steps` steps of x from `x0` on, saving the carry-in or not.
        let run = |x0: u64, steps: usize, stride: usize, state: &[f32], save: bool| {
            let u = GpuBuffer::zeros(st, steps * di).expect("u");
            let u_t = GpuBuffer::zeros(st, di * stride).expect("u_t");
            poison(st, u.cached_ptr(), steps * di * 4);
            poison(st, u_t.cached_ptr(), di * stride * 4);
            let h = floats(&ctx, state);
            let saved = GpuBuffer::zeros(st, di * dc).expect("saved");
            let ints = [
                Arg::I(1),
                Arg::I(steps as i32),
                Arg::I(di as i32),
                Arg::I(dc as i32),
                Arg::I(xs as i32),
            ];
            let tail = [
                Arg::P(x0),
                Arg::P(w.cached_ptr()),
                Arg::P(bias.cached_ptr()),
            ];
            if dc > 8 {
                let cfg = LaunchConfig {
                    grid_dim: (1, steps.div_ceil(32) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut args = vec![
                    Arg::P(u.cached_ptr()),
                    Arg::P(if save { 0 } else { u_t.cached_ptr() }),
                    Arg::I(stride as i32),
                    Arg::P(h.cached_ptr()),
                    Arg::P(if save { saved.cached_ptr() } else { 0 }),
                ];
                args.extend(tail);
                args.extend(ints);
                args.push(Arg::I(32));
                launch_runs(&ctx, k.conv1d_fwd_any_typed.get(width).get(dt), cfg, &args);
            } else if save {
                let mut args = vec![
                    Arg::P(u.cached_ptr()),
                    Arg::P(h.cached_ptr()),
                    Arg::P(saved.cached_ptr()),
                ];
                args.extend(tail);
                args.extend(ints);
                launch_runs(
                    &ctx,
                    k.conv1d_burnin_fwd_tiled_typed.get(width).get(dt),
                    grid_conv_fwd_tiled(1, di, steps).unwrap(),
                    &args,
                );
            } else {
                let mut args = vec![
                    Arg::P(u.cached_ptr()),
                    Arg::P(u_t.cached_ptr()),
                    Arg::P(h.cached_ptr()),
                ];
                args.extend(tail);
                args.extend(ints);
                args.push(Arg::I(stride as i32));
                launch_runs(
                    &ctx,
                    k.conv1d_burnin_nosave_tiled_typed.get(width).get(dt),
                    grid_conv_fwd_tiled(1, di, steps).unwrap(),
                    &args,
                );
            }
            (
                read(st, u.cached_ptr(), steps * di * 4),
                read(st, u_t.cached_ptr(), di * stride * 4),
                read(st, h.cached_ptr(), di * dc * 4),
            )
        };
        for save in [true, false] {
            let big = run(x.cached_ptr(), t, ts, &state0, save);
            let own = run(
                x.cached_ptr() + (t1 * xs * 4) as u64,
                tw,
                tws,
                &window_state,
                save,
            );
            let what = format!("d_conv {dc}, saving {save}");
            assert!(big.0[t1 * di * 4..] == own.0[..], "{what}: outputs");
            assert!(big.2 == own.2, "{what}: carry-out");
            if !save {
                for c in 0..di {
                    let at = (c * ts + t1) * 4;
                    assert!(
                        big.1[at..at + tw * 4] == own.1[c * tws * 4..(c * tws + tw) * 4],
                        "{what}: t-major copy of channel {c}"
                    );
                }
            }
        }
    }
}

/// Steps whose tiles of 128 pass the grid's y extent. The backward conv
/// kernels, the tiled one up to eight taps and the one of any width past
/// them, launch their tiles in runs, and the steps of the second run come
/// out exactly as a launch over just those steps gives them, its carry-in
/// the x rows before them: the x gradient and every tile's tap and bias
/// partial row.
#[test]
fn the_conv_backward_gives_tiles_past_the_y_extent_their_own_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let k = &ctx.kernels;
    let dt = WeightDtype::F32;
    let (di, xs) = (8usize, 16usize);
    let t1 = RUN * 128;
    let t = t1 + 205;
    let tw = t - t1;
    assert!(grid_conv_tiled(1, di, t).unwrap().grid_dim.1 as usize > RUN);
    let width = conv_index_width(1, t, di, 12, xs);
    assert_eq!(width, IndexWidth::I32);
    let (ost, ooff) = (2 * di, di);
    for dc in [4usize, 12] {
        let x_host = det(t * xs, 80 + dc as u32, 0.6);
        let x = floats(&ctx, &x_host);
        let d_u = floats(&ctx, &det(t * di, 81, 0.5));
        let w = floats(&ctx, &det(di * dc, 82, 0.4));
        let bias = floats(&ctx, &det(di, 83, 0.1));
        let init0 = det(di * dc, 84, 0.3);
        let window_init = carry_in(&x_host, xs, di, dc, t1);
        // (x gradient rows, tap partials, bias partials) of one launch over
        // `steps` steps of d_u and x from `first` on.
        let run = |first: usize, steps: usize, init: &[f32]| {
            let tiles = steps.div_ceil(128);
            let d_x = GpuBuffer::zeros(st, steps * ost).expect("d_x");
            let pw = GpuBuffer::zeros(st, tiles * di * dc).expect("tap partials");
            let pb = GpuBuffer::zeros(st, tiles * di).expect("bias partials");
            poison(st, d_x.cached_ptr(), steps * ost * 4);
            poison(st, pw.cached_ptr(), tiles * di * dc * 4);
            poison(st, pb.cached_ptr(), tiles * di * 4);
            let conv_init = floats(&ctx, init);
            let (du0, x0) = (
                d_u.cached_ptr() + (first * di * 4) as u64,
                x.cached_ptr() + (first * xs * 4) as u64,
            );
            let ints = [
                Arg::I(1),
                Arg::I(steps as i32),
                Arg::I(di as i32),
                Arg::I(dc as i32),
            ];
            let outs = [
                Arg::P(d_x.cached_ptr()),
                Arg::P(pw.cached_ptr()),
                Arg::P(pb.cached_ptr()),
            ];
            let strides = [Arg::I(ost as i32), Arg::I(ooff as i32), Arg::I(xs as i32)];
            if dc > 8 {
                let dco = GpuBuffer::zeros(st, steps * di).expect("dco");
                let mut args = vec![
                    Arg::P(dco.cached_ptr()),
                    Arg::P(du0),
                    Arg::P(x0),
                    Arg::P(conv_init.cached_ptr()),
                    Arg::P(w.cached_ptr()),
                    Arg::P(bias.cached_ptr()),
                ];
                args.extend(ints);
                args.push(Arg::I(xs as i32));
                launch(
                    &ctx,
                    k.conv1d_bwd_dco_typed.get(dt),
                    grid_1d_wide(steps * di),
                    &args,
                );
                let mut args = outs.to_vec();
                args.extend([
                    Arg::P(dco.cached_ptr()),
                    Arg::P(x0),
                    Arg::P(conv_init.cached_ptr()),
                    Arg::P(w.cached_ptr()),
                ]);
                args.extend(ints);
                args.extend(strides);
                launch_runs(
                    &ctx,
                    k.conv1d_bwd_any_typed.get(width).get(dt),
                    grid_conv_tiled(1, di, steps).unwrap(),
                    &args,
                );
            } else {
                let mut args = outs.to_vec();
                args.extend([
                    Arg::P(du0),
                    Arg::P(x0),
                    Arg::P(conv_init.cached_ptr()),
                    Arg::P(w.cached_ptr()),
                    Arg::P(bias.cached_ptr()),
                ]);
                args.extend(ints);
                args.extend(strides);
                launch_runs(
                    &ctx,
                    k.conv1d_bwd_tiled_typed.get(width).get(dt),
                    grid_conv_tiled(1, di, steps).unwrap(),
                    &args,
                );
            }
            (
                read(st, d_x.cached_ptr(), steps * ost * 4),
                read(st, pw.cached_ptr(), tiles * di * dc * 4),
                read(st, pb.cached_ptr(), tiles * di * 4),
            )
        };
        let big = run(0, t, &init0);
        let own = run(t1, tw, &window_init);
        assert!(
            big.0[t1 * ost * 4..] == own.0[..],
            "d_conv {dc}: x gradient"
        );
        let first_tile = t1 / 128;
        assert!(
            big.1[first_tile * di * dc * 4..] == own.1[..],
            "d_conv {dc}: tap partials"
        );
        assert!(
            big.2[first_tile * di * 4..] == own.2[..],
            "d_conv {dc}: bias partials"
        );
    }
}

/// The t-major transposes, the softplus into t-major rows and the gate out
/// of them, over steps whose tiles of 32 pass the grid's y extent and over
/// sequences past its z extent: the rows of the later runs come out
/// exactly as a launch over just those rows gives them.
#[test]
fn the_tmajor_transposes_give_rows_past_the_y_and_z_extents_their_own_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let k = &ctx.kernels;
    let dt = WeightDtype::Bf16;
    let es = dt.size_bytes();
    let d = 32usize;
    let gs = 2 * d;
    // (sequences, steps, first sequence, first step) of each case: the
    // steps past the y extent in one sequence, then the sequences past the
    // z extent.
    let t1 = RUN * 32;
    for (b, t, b1, t1) in [(1usize, t1 + 37, 0usize, t1), (RUN + 2, 32, RUN, 0)] {
        let cfg = grid_tmajor_tiles(b, t, d);
        assert!(cfg.grid_dim.1 as usize > RUN || cfg.grid_dim.2 as usize > RUN);
        let (bw, tw) = (b - b1, t - t1);
        let ts = t.next_multiple_of(8);
        let src = typed(&ctx, &det(b * t * d, 90, 4.0), dt);
        let gate = typed(&ctx, &det(b * t * gs, 91, 2.0), dt);
        // The first row of the window in [b, t, cols] rows and in the
        // t-major [b, d, ts] rows.
        let row = |cols: usize| ((b1 * t + t1) * cols * es) as u64;
        let tmajor_at = ((b1 * d * ts + t1) * es) as u64;
        // One softplus into t-major rows of stride `ts` from `src0`, then
        // the gate back out of them, over `nb` sequences of `nt` steps.
        let run = |src0: u64, gate0: u64, nb: usize, nt: usize| {
            let y_t = DtypedBuf::zeros(st, nb * d * ts, dt).expect("y_t");
            let out = DtypedBuf::zeros(st, nb * nt * d, dt).expect("out");
            poison(st, y_t.cached_ptr(), nb * d * ts * es);
            poison(st, out.cached_ptr(), nb * nt * d * es);
            launch_yz_runs(
                &ctx,
                k.softplus_fwd_tmajor.get(dt),
                grid_tmajor_tiles(nb, nt, d),
                &[
                    Arg::P(y_t.cached_ptr()),
                    Arg::P(src0),
                    Arg::I(nt as i32),
                    Arg::I(d as i32),
                    Arg::I(ts as i32),
                ],
            );
            launch_yz_runs(
                &ctx,
                k.gate_from_tmajor.get(dt),
                grid_tmajor_tiles(nb, nt, d),
                &[
                    Arg::P(out.cached_ptr()),
                    Arg::P(y_t.cached_ptr()),
                    Arg::P(gate0),
                    Arg::I(nt as i32),
                    Arg::I(d as i32),
                    Arg::I(ts as i32),
                    Arg::I(gs as i32),
                ],
            );
            (
                read(st, y_t.cached_ptr(), nb * d * ts * es),
                read(st, out.cached_ptr(), nb * nt * d * es),
            )
        };
        let big = run(src.cached_ptr(), gate.cached_ptr(), b, t);
        let own = run(
            src.cached_ptr() + row(d),
            gate.cached_ptr() + row(gs),
            bw,
            tw,
        );
        let what = format!("{b} sequences of {t} steps");
        assert!(big.1[row(d) as usize..] == own.1[..], "{what}: gated rows");
        // The window's t-major rows keep the stride; the big launch's
        // sit `t1` steps into each of its own.
        for r in 0..bw * d {
            let at = tmajor_at as usize + r * ts * es;
            assert!(
                big.0[at..at + tw * es] == own.0[r * ts * es..(r * ts + tw) * es],
                "{what}: t-major row {r}"
            );
        }
    }
}

/// The staged t-major B and C gather over sequences past the grid's y
/// extent, f32 and bf16: the sequences of the second run come out exactly
/// as a launch over just them gives them.
#[test]
fn the_tiled_bc_gather_gives_sequences_past_the_y_extent_their_own_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let k = &ctx.kernels;
    let (b, t, stride, ds) = (RUN + 2, 32usize, 40usize, 16usize);
    let (b_off, c_off) = (8i32, 24i32);
    let b1 = RUN;
    for dt in [WeightDtype::F32, WeightDtype::Bf16] {
        let es = dt.size_bytes();
        let src = typed(&ctx, &det(b * t * stride, 95, 1.0), dt);
        let f = k.gather_bc_cols_tmajor_tiled_typed.get(dt);
        let run = |src0: u64, nb: usize| {
            let n = nb * ds * t;
            let (bo, co) = (
                DtypedBuf::zeros(st, n, dt).expect("b"),
                DtypedBuf::zeros(st, n, dt).expect("c"),
            );
            poison(st, bo.cached_ptr(), n * es);
            poison(st, co.cached_ptr(), n * es);
            launch_runs(
                &ctx,
                f,
                grid_gather_bc_tiled(nb, t, ds, es),
                &[
                    Arg::P(bo.cached_ptr()),
                    Arg::P(co.cached_ptr()),
                    Arg::P(src0),
                    Arg::I(t as i32),
                    Arg::I(stride as i32),
                    Arg::I(ds as i32),
                    Arg::I(b_off),
                    Arg::I(c_off),
                ],
            );
            (
                read(st, bo.cached_ptr(), n * es),
                read(st, co.cached_ptr(), n * es),
            )
        };
        let big = run(src.cached_ptr(), b);
        let own = run(src.cached_ptr() + (b1 * t * stride * es) as u64, b - b1);
        let at = b1 * ds * t * es;
        assert!(big.0[at..] == own.0[..], "{dt:?}: B rows");
        assert!(big.1[at..] == own.1[..], "{dt:?}: C rows");
    }
}

/// The staged t-major B and C gather at every kind of state width: inside
/// one slice of states, a whole slice, past it by one, the width the
/// unsliced tile could no longer stage in 48 KB (187), and several slices
/// with a ragged last one, over windows that end inside a token tile, in
/// f32, bf16 and f16. Every value lands at its t-major place, bit for bit
/// the x projection's.
#[test]
fn the_tiled_bc_gather_writes_every_state_width_t_major() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let k = &ctx.kernels;
    let b_off = 3usize;
    let mut cases = 0;
    for (b, t, ds) in [
        (1usize, 1usize, 1usize),
        (2, 33, 16),
        (2, 70, 64),
        (3, 31, 65),
        (1, 100, 187),
        (2, 45, 300),
    ] {
        let stride = b_off + 2 * ds + 5;
        for dt in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
            let es = dt.size_bytes();
            let src = typed(&ctx, &det(b * t * stride, 96, 1.0), dt);
            let n = b * ds * t;
            let (bo, co) = (
                DtypedBuf::zeros(st, n, dt).expect("b"),
                DtypedBuf::zeros(st, n, dt).expect("c"),
            );
            poison(st, bo.cached_ptr(), n * es);
            poison(st, co.cached_ptr(), n * es);
            launch_runs(
                &ctx,
                k.gather_bc_cols_tmajor_tiled_typed.get(dt),
                grid_gather_bc_tiled(b, t, ds, es),
                &[
                    Arg::P(bo.cached_ptr()),
                    Arg::P(co.cached_ptr()),
                    Arg::P(src.cached_ptr()),
                    Arg::I(t as i32),
                    Arg::I(stride as i32),
                    Arg::I(ds as i32),
                    Arg::I(b_off as i32),
                    Arg::I((b_off + ds) as i32),
                ],
            );
            let rows = read(st, src.cached_ptr(), b * t * stride * es);
            let (gb, gc) = (
                read(st, bo.cached_ptr(), n * es),
                read(st, co.cached_ptr(), n * es),
            );
            for (bb, nn, tt) in
                (0..b).flat_map(|bb| (0..ds).flat_map(move |nn| (0..t).map(move |tt| (bb, nn, tt))))
            {
                let dst = ((bb * ds + nn) * t + tt) * es;
                let row = ((bb * t + tt) * stride + b_off + nn) * es;
                assert!(
                    gb[dst..dst + es] == rows[row..row + es]
                        && gc[dst..dst + es] == rows[row + ds * es..row + (ds + 1) * es],
                    "{dt:?} B {b} T {t} ds {ds}: sequence {bb} state {nn} token {tt}"
                );
            }
            cases += 1;
        }
    }
    eprintln!("t-major B/C gather: every value in place on {cases} cases");
}

/// The per-sequence column sum over sequences past the grid's y extent:
/// the sums of the second run come out exactly as a launch over just those
/// sequences gives them.
#[test]
fn the_sequence_column_sum_gives_sequences_past_the_y_extent_their_own_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let (segs, len, cols) = (RUN + 2, 3usize, 40usize);
    let s1 = RUN;
    let src = floats(&ctx, &det(segs * len * cols, 97, 1.0));
    let run = |src0: u64, n: usize| {
        let out = GpuBuffer::zeros(st, n * cols).expect("out");
        poison(st, out.cached_ptr(), n * cols * 4);
        let cfg = LaunchConfig {
            grid_dim: ((cols as u32).div_ceil(256), n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_runs(
            &ctx,
            &ctx.kernels.colsum_segments,
            cfg,
            &[
                Arg::P(out.cached_ptr()),
                Arg::P(src0),
                Arg::I(n as i32),
                Arg::I(len as i32),
                Arg::I(cols as i32),
            ],
        );
        read(st, out.cached_ptr(), n * cols * 4)
    };
    let big = run(src.cached_ptr(), segs);
    let own = run(src.cached_ptr() + (s1 * len * cols * 4) as u64, segs - s1);
    assert!(big[s1 * cols * 4..] == own[..], "sums past the extent");
}

/// Whether, in `bytes` of `outer` blocks of `channels` rows of `row`
/// elements of `es` bytes each, every row `c < pairs` equals row
/// `c + shift` of the same block.
fn pairs_match(
    bytes: &[u8],
    es: usize,
    (outer, channels, row): (usize, usize, usize),
    pairs: usize,
    shift: usize,
) -> bool {
    assert_eq!(bytes.len(), outer * channels * row * es);
    (0..outer).all(|o| {
        (0..pairs).all(|c| {
            let at = |c: usize| ((o * channels + c) * row) * es;
            bytes[at(c)..at(c) + row * es] == bytes[at(c + shift)..at(c + shift) + row * es]
        })
    })
}

/// Copy each listed row group of `v` to the same group `shift` rows on:
/// `v` holds `outer` blocks of `channels` rows of `row` values.
fn pair_up(
    v: &mut [f32],
    (outer, channels, row): (usize, usize, usize),
    pairs: usize,
    shift: usize,
) {
    for o in 0..outer {
        for c in 0..pairs {
            let from = (o * channels + c) * row;
            let to = (o * channels + c + shift) * row;
            v.copy_within(from..from + row, to);
        }
    }
}

/// The parallel scans over more channels than the grid's y extent holds:
/// the saving forward, the forward without saves, the ungrouped backward
/// and the fold, whose groups of four also pass the extent. Channels
/// SHIFT on from the first PAIRS share every input with them, so each must
/// come out with the bits of its twin in the first run: outputs, final
/// state, tape and every gradient row.
#[test]
fn the_parallel_scans_give_channels_past_the_y_extent_their_own_bits() {
    const PAIRS: usize = 64;
    const SHIFT: usize = 4 * 65536;
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let k = &ctx.kernels;
    let dt = WeightDtype::Bf16;
    let es = dt.size_bytes();
    let (b, t, di, ds) = (1usize, 64usize, SHIFT + PAIRS, 16usize);
    assert!(di / SCAN_BWD_DGROUP > RUN);
    let act = (t, di, 1);
    let rows = |row: usize| (1, di, row);
    let n_chunks = t.div_ceil(mamba_rs::gpu::launch::SCAN_CHUNK);
    let tape_len = scan_tape_len(b, t, di, ds);
    assert_eq!(tape_len, di * ds * 3 * n_chunks);

    let mut delta_raw = det(t * di, 100, 2.0);
    let mut u = det(t * di, 101, 1.0);
    let mut dy = det(t * di, 102, 0.2);
    let mut a = det(di * ds, 103, 0.6)
        .iter()
        .map(|x| -(x.abs() + 0.05))
        .collect::<Vec<_>>();
    let mut dp = det(di, 104, 1.0);
    let mut h0 = det(di * ds, 105, 0.8);
    for (v, shape) in [
        (&mut delta_raw, act),
        (&mut u, act),
        (&mut dy, act),
        (&mut a, rows(ds)),
        (&mut dp, rows(1)),
        (&mut h0, rows(ds)),
    ] {
        pair_up(v, shape, PAIRS, SHIFT);
    }
    let delta_raw = typed(&ctx, &delta_raw, dt);
    let u = typed(&ctx, &u, dt);
    let dy = typed(&ctx, &dy, dt);
    let a = floats(&ctx, &a);
    let dp = floats(&ctx, &dp);
    let bm = typed(&ctx, &det(b * ds * t, 106, 0.7), dt);
    let cm = typed(&ctx, &det(b * ds * t, 107, 0.7), dt);
    let ints = [
        Arg::I(b as i32),
        Arg::I(t as i32),
        Arg::I(di as i32),
        Arg::I(ds as i32),
    ];

    // The saving forward over the slim tape, as the trainer runs it.
    let h = floats(&ctx, &h0);
    let y = DtypedBuf::zeros(st, t * di, dt).expect("y");
    let delta = DtypedBuf::zeros(st, t * di, dt).expect("delta");
    let tape = GpuBuffer::zeros(st, tape_len).expect("tape");
    poison(st, y.cached_ptr(), t * di * es);
    poison(st, delta.cached_ptr(), t * di * es);
    poison(st, tape.cached_ptr(), tape_len * 4);
    let mut args = vec![
        Arg::P(h.cached_ptr()),
        Arg::P(y.cached_ptr()),
        Arg::P(tape.cached_ptr()),
        Arg::P(delta_raw.cached_ptr()),
        Arg::P(delta.cached_ptr()),
        Arg::P(u.cached_ptr()),
        Arg::P(bm.cached_ptr()),
        Arg::P(cm.cached_ptr()),
        Arg::P(a.cached_ptr()),
        Arg::P(dp.cached_ptr()),
    ];
    args.extend(ints);
    args.extend([Arg::P(tape.cached_ptr()), Arg::I(1)]);
    let width = scan_fwd_index_width(b, t, di, ds, true);
    launch_runs(
        &ctx,
        k.ssm_parallel_fwd_typed.get(width).get(dt),
        grid_parallel_scan(b, di, ds),
        &args,
    );
    let tape_row = ds * 3 * n_chunks;
    for (what, ptr, bytes, shape, e) in [
        ("y", y.cached_ptr(), t * di * es, act, es),
        ("delta", delta.cached_ptr(), t * di * es, act, es),
        ("state", h.cached_ptr(), di * ds * 4, rows(ds), 4),
        ("tape", tape.cached_ptr(), tape_len * 4, rows(tape_row), 4),
    ] {
        assert!(
            pairs_match(&read(st, ptr, bytes), e, shape, PAIRS, SHIFT),
            "saving forward: {what}"
        );
    }

    // The forward without saves, from the same state, on the post-softplus
    // delta the saving forward stored.
    let h2 = floats(&ctx, &h0);
    let y2 = DtypedBuf::zeros(st, t * di, dt).expect("y");
    poison(st, y2.cached_ptr(), t * di * es);
    let mut args = vec![
        Arg::P(h2.cached_ptr()),
        Arg::P(y2.cached_ptr()),
        Arg::P(delta.cached_ptr()),
        Arg::P(u.cached_ptr()),
        Arg::P(bm.cached_ptr()),
        Arg::P(cm.cached_ptr()),
        Arg::P(a.cached_ptr()),
        Arg::P(dp.cached_ptr()),
        Arg::P(0),
        Arg::I(0),
    ];
    args.extend(ints);
    args.push(Arg::I(0));
    let width = scan_fwd_nosave_index_width(b, t, di, ds, 2 * di, false);
    launch_runs(
        &ctx,
        k.ssm_parallel_fwd_nosave_typed.get(width).get(dt),
        grid_parallel_scan(b, di, ds),
        &args,
    );
    assert!(
        pairs_match(
            &read(st, y2.cached_ptr(), t * di * es),
            es,
            act,
            PAIRS,
            SHIFT
        ),
        "forward without saves: y"
    );
    assert!(
        pairs_match(
            &read(st, h2.cached_ptr(), di * ds * 4),
            4,
            rows(ds),
            PAIRS,
            SHIFT
        ),
        "forward without saves: state"
    );

    // Both backwards over the slim tape.
    for fold in [false, true] {
        let g = if fold { SCAN_BWD_DGROUP } else { 1 };
        let grad_rows = di / g;
        let a_rows = if fold { n_chunks } else { 1 };
        let d_delta = DtypedBuf::zeros(st, t * di, dt).expect("d_delta");
        let d_u = DtypedBuf::zeros(st, t * di, dt).expect("d_u");
        let d_b = DtypedBuf::zeros(st, ds * grad_rows * t, dt).expect("d_b");
        let d_c = DtypedBuf::zeros(st, ds * grad_rows * t, dt).expect("d_c");
        let d_d = GpuBuffer::zeros(st, di).expect("d_d");
        let d_a = GpuBuffer::zeros(st, a_rows * di * ds).expect("d_a");
        for (ptr, bytes) in [
            (d_delta.cached_ptr(), t * di * es),
            (d_u.cached_ptr(), t * di * es),
            (d_b.cached_ptr(), ds * grad_rows * t * es),
            (d_c.cached_ptr(), ds * grad_rows * t * es),
            (d_d.cached_ptr(), di * 4),
        ] {
            poison(st, ptr, bytes);
        }
        // The ungrouped backward adds into its d_a_log rows; the fold
        // assigns every slot.
        if fold {
            poison(st, d_a.cached_ptr(), a_rows * di * ds * 4);
        }
        let mut args = vec![
            Arg::P(tape.cached_ptr()),
            Arg::P(delta.cached_ptr()),
            Arg::P(u.cached_ptr()),
            Arg::P(bm.cached_ptr()),
            Arg::P(cm.cached_ptr()),
            Arg::P(a.cached_ptr()),
            Arg::P(dp.cached_ptr()),
            Arg::P(dy.cached_ptr()),
            Arg::P(d_delta.cached_ptr()),
        ];
        if fold {
            args.push(Arg::P(delta_raw.cached_ptr()));
        }
        args.extend([
            Arg::P(d_u.cached_ptr()),
            Arg::P(d_b.cached_ptr()),
            Arg::P(d_c.cached_ptr()),
            Arg::P(d_d.cached_ptr()),
            Arg::P(d_a.cached_ptr()),
        ]);
        args.extend(ints);
        args.extend([Arg::P(tape.cached_ptr()), Arg::I(1)]);
        let width = scan_bwd_index_width(b, t, di, ds, true, fold);
        let (f, cfg) = if fold {
            (
                k.ssm_parallel_bwd_fold_typed.get(width).get(dt),
                grid_parallel_scan_bwd_fold(b, di, ds, es),
            )
        } else {
            (
                k.ssm_parallel_bwd_typed.get(width).get(dt),
                grid_parallel_scan_bwd(b, di),
            )
        };
        launch_runs(&ctx, f, cfg, &args);
        let gpairs = (PAIRS / g, SHIFT / g);
        for (what, ptr, bytes, shape, e, (pairs, shift)) in [
            (
                "d_delta",
                d_delta.cached_ptr(),
                t * di * es,
                act,
                es,
                (PAIRS, SHIFT),
            ),
            (
                "d_u",
                d_u.cached_ptr(),
                t * di * es,
                act,
                es,
                (PAIRS, SHIFT),
            ),
            (
                "d_B",
                d_b.cached_ptr(),
                ds * grad_rows * t * es,
                (ds, grad_rows, t),
                es,
                gpairs,
            ),
            (
                "d_C",
                d_c.cached_ptr(),
                ds * grad_rows * t * es,
                (ds, grad_rows, t),
                es,
                gpairs,
            ),
            ("d_D", d_d.cached_ptr(), di * 4, rows(1), 4, (PAIRS, SHIFT)),
            (
                "d_a_log",
                d_a.cached_ptr(),
                a_rows * di * ds * 4,
                (a_rows, di, ds),
                4,
                (PAIRS, SHIFT),
            ),
        ] {
            assert!(
                pairs_match(&read(st, ptr, bytes), e, shape, pairs, shift),
                "backward (fold {fold}): {what}"
            );
        }
    }
}
