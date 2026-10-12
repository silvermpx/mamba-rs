//! The Mamba-1 conv kernels of any width against the tiled kernels that
//! hold the window in registers, at widths both serve: the training
//! forward (outputs, carry-out, saved window), the prefill forward with
//! its t-major copy, and the backward (the x gradient into a strided
//! destination, the tap and bias partials), in every activation type.
//! They run the same chains in the same order, so every output keeps its
//! bits.
#![cfg(feature = "cuda")]

use cudarc::driver::PushKernelArg;
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{grid_1d, grid_conv_fwd_tiled, grid_conv_tiled};

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

const DTYPES: [WeightDtype; 3] = [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16];

/// A kernel argument: a device pointer or an int.
enum Arg {
    P(u64),
    I(i32),
}

fn launch(
    ctx: &GpuCtx,
    f: &cudarc::driver::CudaFunction,
    cfg: cudarc::driver::LaunchConfig,
    args: &[Arg],
) {
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
    unsafe { bld.launch(cfg) }.unwrap();
}

fn bits_f32(ctx: &GpuCtx, b: &GpuBuffer) -> Vec<u32> {
    ctx.stream.synchronize().unwrap();
    b.to_cpu(&ctx.stream)
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn bits_typed(ctx: &GpuCtx, b: &DtypedBuf, n: usize) -> Vec<u32> {
    ctx.stream.synchronize().unwrap();
    let mut v = vec![0f32; n];
    b.download_f32(&ctx.stream, &mut v).unwrap();
    ctx.stream.synchronize().unwrap();
    v.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn m1_conv_of_any_width_is_the_tiled_conv() {
    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new_with_state_cap(&device, 16).unwrap();
    let k = &ctx.kernels;
    let s = &ctx.stream;
    // 150 steps: several forward and two backward tiles. x is read through
    // a row stride of 2 * d_inner, as from the in_proj output.
    let (b, t, di) = (2usize, 150usize, 24usize);
    let (bt, xs) = (b * t, 2 * di);
    for dc in [1usize, 3, 4, 8] {
        for dtype in DTYPES {
            let typed = |data: &[f32]| -> DtypedBuf {
                let buf = DtypedBuf::zeros(s, data.len(), dtype).unwrap();
                s.synchronize().unwrap();
                buf.upload_f32(s, data).unwrap();
                s.synchronize().unwrap();
                buf
            };
            let f32b = |data: &[f32]| -> GpuBuffer {
                let mut buf = GpuBuffer::zeros(s, data.len()).unwrap();
                s.synchronize().unwrap();
                buf.upload(s, data).unwrap();
                s.synchronize().unwrap();
                buf
            };
            let x = typed(&det(bt * xs, 31, 0.5));
            let d_u = typed(&det(bt * di, 32, 0.1));
            let weight = f32b(&det(di * dc, 33, 0.4));
            let bias = f32b(&det(di, 34, 0.1));
            let state0 = det(b * di * dc, 35, 0.3);
            let (bi, ti, dii, dci, xsi) = (b as i32, t as i32, di as i32, dc as i32, xs as i32);
            let (w, bs) = (weight.cached_ptr(), bias.cached_ptr());

            // The training forward: outputs, carry-out, saved window.
            let train = |any: bool| -> Vec<Vec<u32>> {
                let u = DtypedBuf::zeros(s, bt * di, dtype).unwrap();
                let state = f32b(&state0);
                let saved = GpuBuffer::zeros(s, b * di * dc).unwrap();
                let common = [
                    Arg::P(state.cached_ptr()),
                    Arg::P(saved.cached_ptr()),
                    Arg::P(x.cached_ptr()),
                    Arg::P(w),
                    Arg::P(bs),
                    Arg::I(bi),
                    Arg::I(ti),
                    Arg::I(dii),
                    Arg::I(dci),
                    Arg::I(xsi),
                ];
                if any {
                    let mut args = vec![Arg::P(u.cached_ptr()), Arg::P(0), Arg::I(0)];
                    args.extend(common);
                    args.push(Arg::I(32));
                    launch(
                        &ctx,
                        k.conv1d_fwd_any_typed.get(dtype),
                        grid_conv_fwd_tiled(b, di, t),
                        &args,
                    );
                } else {
                    let mut args = vec![Arg::P(u.cached_ptr())];
                    args.extend(common);
                    launch(
                        &ctx,
                        k.conv1d_burnin_fwd_tiled_typed.get(dtype),
                        grid_conv_fwd_tiled(b, di, t),
                        &args,
                    );
                }
                vec![
                    bits_typed(&ctx, &u, bt * di),
                    bits_f32(&ctx, &state),
                    bits_f32(&ctx, &saved),
                ]
            };
            let (want, got) = (train(false), train(true));
            for (i, name) in ["outputs", "carry-out", "saved window"].iter().enumerate() {
                assert!(
                    got[i] == want[i],
                    "{dtype:?} d_conv {dc}: training forward {name}"
                );
            }

            // The prefill forward with its t-major copy (rows of 152).
            let uts = t.next_multiple_of(8);
            let prefill = |any: bool| -> Vec<Vec<u32>> {
                let u = DtypedBuf::zeros(s, bt * di, dtype).unwrap();
                let u_t = DtypedBuf::zeros(s, b * di * uts, dtype).unwrap();
                let state = f32b(&state0);
                let (up, utp, sp) = (u.cached_ptr(), u_t.cached_ptr(), state.cached_ptr());
                let tail = [
                    Arg::P(x.cached_ptr()),
                    Arg::P(w),
                    Arg::P(bs),
                    Arg::I(bi),
                    Arg::I(ti),
                    Arg::I(dii),
                    Arg::I(dci),
                    Arg::I(xsi),
                ];
                if any {
                    let mut args = vec![
                        Arg::P(up),
                        Arg::P(utp),
                        Arg::I(uts as i32),
                        Arg::P(sp),
                        Arg::P(0),
                    ];
                    args.extend(tail);
                    args.push(Arg::I(32));
                    launch(
                        &ctx,
                        k.conv1d_fwd_any_typed.get(dtype),
                        grid_conv_fwd_tiled(b, di, t),
                        &args,
                    );
                } else {
                    let mut args = vec![Arg::P(up), Arg::P(utp), Arg::P(sp)];
                    args.extend(tail);
                    args.push(Arg::I(uts as i32));
                    launch(
                        &ctx,
                        k.conv1d_burnin_nosave_tiled_typed.get(dtype),
                        grid_conv_fwd_tiled(b, di, t),
                        &args,
                    );
                }
                vec![
                    bits_typed(&ctx, &u, bt * di),
                    bits_typed(&ctx, &u_t, b * di * uts),
                    bits_f32(&ctx, &state),
                ]
            };
            let (want, got) = (prefill(false), prefill(true));
            for (i, name) in ["outputs", "t-major outputs", "carry-out"]
                .iter()
                .enumerate()
            {
                assert!(
                    got[i] == want[i],
                    "{dtype:?} d_conv {dc}: prefill forward {name}"
                );
            }

            // The backward into the x half of a [bt, 2 * di] gradient.
            let conv_init = f32b(&state0);
            let n_tiles = t.div_ceil(128);
            let backward = |any: bool| -> Vec<Vec<u32>> {
                let d_x = DtypedBuf::zeros(s, bt * xs, dtype).unwrap();
                let wp = GpuBuffer::zeros(s, b * n_tiles * di * dc).unwrap();
                let bp = GpuBuffer::zeros(s, b * n_tiles * di).unwrap();
                let (dxp, wpp, bpp) = (d_x.cached_ptr(), wp.cached_ptr(), bp.cached_ptr());
                let (xp, cip) = (x.cached_ptr(), conv_init.cached_ptr());
                let tail = [
                    Arg::I(bi),
                    Arg::I(ti),
                    Arg::I(dii),
                    Arg::I(dci),
                    Arg::I(xsi),
                    Arg::I(0),
                    Arg::I(xsi),
                ];
                if any {
                    let dco = GpuBuffer::zeros(s, bt * di).unwrap();
                    let dcop = dco.cached_ptr();
                    launch(
                        &ctx,
                        k.conv1d_bwd_dco_typed.get(dtype),
                        grid_1d(bt * di),
                        &[
                            Arg::P(dcop),
                            Arg::P(d_u.cached_ptr()),
                            Arg::P(xp),
                            Arg::P(cip),
                            Arg::P(w),
                            Arg::P(bs),
                            Arg::I(bi),
                            Arg::I(ti),
                            Arg::I(dii),
                            Arg::I(dci),
                            Arg::I(xsi),
                        ],
                    );
                    let mut args = vec![
                        Arg::P(dxp),
                        Arg::P(wpp),
                        Arg::P(bpp),
                        Arg::P(dcop),
                        Arg::P(xp),
                        Arg::P(cip),
                        Arg::P(w),
                    ];
                    args.extend(tail);
                    launch(
                        &ctx,
                        k.conv1d_bwd_any_typed.get(dtype),
                        grid_conv_tiled(b, di, t),
                        &args,
                    );
                } else {
                    let mut args = vec![
                        Arg::P(dxp),
                        Arg::P(wpp),
                        Arg::P(bpp),
                        Arg::P(d_u.cached_ptr()),
                        Arg::P(xp),
                        Arg::P(cip),
                        Arg::P(w),
                        Arg::P(bs),
                    ];
                    args.extend(tail);
                    launch(
                        &ctx,
                        k.conv1d_bwd_tiled_typed.get(dtype),
                        grid_conv_tiled(b, di, t),
                        &args,
                    );
                }
                vec![
                    bits_typed(&ctx, &d_x, bt * xs),
                    bits_f32(&ctx, &wp),
                    bits_f32(&ctx, &bp),
                ]
            };
            let (want, got) = (backward(false), backward(true));
            for (i, name) in ["x gradient", "tap partials", "bias partials"]
                .iter()
                .enumerate()
            {
                assert!(got[i] == want[i], "{dtype:?} d_conv {dc}: backward {name}");
            }
        }
    }
}
