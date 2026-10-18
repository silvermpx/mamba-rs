//! The two builds of the Mamba-1 sequential scans and the parallel scan
//! backward. Each is compiled with `int` offsets and with `long long`
//! offsets (the `_i64` entry), and a launch takes the `int` build while
//! every offset it forms fits the 32-bit range. The two builds give the
//! same bits on every activation type and route, and past 2^31 the fold's
//! `_i64` build gives every sequence of a batch exactly what that sequence
//! gives alone.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::sys::CUresult::CUDA_SUCCESS;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{
    IndexWidth, SCAN_BWD_DGROUP, SCAN_CHUNK, grid_1d_cover, grid_parallel_scan_bwd,
    grid_parallel_scan_bwd_fold, grid_seq, scan_bwd_fold_carry_len, scan_bwd_index_width,
    seq_lanes, y_runs,
};

const DTYPES: [WeightDtype; 3] = [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16];
const WIDTHS: [IndexWidth; 2] = [IndexWidth::I32, IndexWidth::I64];

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

fn a_neg(n: usize, seed: u32) -> Vec<f32> {
    det(n, seed, 0.6)
        .iter()
        .map(|x| -(x.abs() + 0.05))
        .collect()
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
    unsafe { bld.launch(cfg) }.expect("scan launch");
}

/// [`launch`] once per run of the grid's y axis, each run's first block
/// appended as the kernel's last argument.
fn launch_runs(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    for (y_first, cfg) in y_runs(cfg) {
        let mut run: Vec<Arg> = args.to_vec();
        run.push(Arg::I(y_first));
        launch(ctx, f, cfg, &run);
    }
}

/// [`launch_runs`] with `tail` after each run's first block: the fold's
/// global carries come last.
fn launch_runs_tail(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg], tail: &[Arg]) {
    for (y_first, cfg) in y_runs(cfg) {
        let mut run = args.to_vec();
        run.push(Arg::I(y_first));
        run.extend_from_slice(tail);
        launch(ctx, f, cfg, &run);
    }
}

/// Every output buffer of a launch, as bytes.
fn bytes_of(ctx: &GpuCtx, outs: &[(u64, usize)]) -> Vec<Vec<u8>> {
    outs.iter().map(|&(p, n)| read(&ctx.stream, p, n)).collect()
}

/// The sequential scans in both offset widths: the saving forward, the
/// gated forward without saves, the decode step and both backwards (the
/// typed one and the f32 one of the f32 route), on a shape of several
/// sequences and a state width that is not a multiple of eight.
#[test]
fn both_builds_of_the_sequential_scans_give_the_same_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let k = &ctx.kernels;
    let (b, t, di, ds) = (3usize, 40usize, 24usize, 13usize);
    let (bt, st) = (b * t, b * di * ds);
    let [bi, ti, dii, dsi] = [b, t, di, ds].map(|v| v as i32);
    let grid = || grid_1d_cover(b * di).expect("grid");
    let seq_grid = || grid_seq(b, di, ds).expect("grid");
    let lanes = seq_lanes(ds) as i32;
    for dt in DTYPES {
        let es = dt.size_bytes();
        let an = floats(&ctx, &a_neg(di * ds, 1));
        let d = floats(&ctx, &det(di, 2, 1.0));
        let x = typed(&ctx, &det(bt * di, 3, 1.0), dt);
        let u = typed(&ctx, &det(bt * di, 4, 1.0), dt);
        let bm = typed(&ctx, &det(bt * ds, 5, 0.7), dt);
        let cm = typed(&ctx, &det(bt * ds, 6, 0.7), dt);
        let gate = typed(&ctx, &det(bt * 2 * di, 7, 2.0), dt);
        let h0 = det(st, 8, 0.8);

        // The saving forward: final state, outputs, the post-softplus save
        // and the h tape [b, t + 1, d_inner, d_state].
        let saving = |w: IndexWidth| -> Vec<Vec<u8>> {
            let h = floats(&ctx, &h0);
            let y = DtypedBuf::zeros(&ctx.stream, bt * di, dt).expect("y");
            let tape = GpuBuffer::zeros(&ctx.stream, b * (t + 1) * di * ds).expect("tape");
            let saved = DtypedBuf::zeros(&ctx.stream, bt * di, dt).expect("delta");
            let args = [
                Arg::P(h.cached_ptr()),
                Arg::P(y.cached_ptr()),
                Arg::P(tape.cached_ptr()),
                Arg::P(x.cached_ptr()),
                Arg::P(saved.cached_ptr()),
                Arg::P(u.cached_ptr()),
                Arg::P(bm.cached_ptr()),
                Arg::P(cm.cached_ptr()),
                Arg::P(an.cached_ptr()),
                Arg::P(d.cached_ptr()),
                Arg::I(bi),
                Arg::I(ti),
                Arg::I(dii),
                Arg::I(dsi),
                Arg::I(lanes),
            ];
            launch(
                &ctx,
                k.ssm_burnin_fwd_typed.get(w).get(dt),
                seq_grid(),
                &args,
            );
            bytes_of(
                &ctx,
                &[
                    (h.cached_ptr(), st * 4),
                    (y.cached_ptr(), bt * di * es),
                    (saved.cached_ptr(), bt * di * es),
                    (tape.cached_ptr(), b * (t + 1) * di * ds * 4),
                ],
            )
        };
        assert!(
            saving(IndexWidth::I32) == saving(IndexWidth::I64),
            "{dt:?} saving forward"
        );

        // The forward without saves, its store gated by the in_proj rows.
        let nosave = |w: IndexWidth| -> Vec<Vec<u8>> {
            let h = floats(&ctx, &h0);
            let y = DtypedBuf::zeros(&ctx.stream, bt * di, dt).expect("y");
            let args = [
                Arg::P(h.cached_ptr()),
                Arg::P(y.cached_ptr()),
                Arg::P(x.cached_ptr()),
                Arg::P(u.cached_ptr()),
                Arg::P(bm.cached_ptr()),
                Arg::P(cm.cached_ptr()),
                Arg::P(an.cached_ptr()),
                Arg::P(d.cached_ptr()),
                Arg::P(gate.cached_ptr()),
                Arg::I(2 * dii),
                Arg::I(bi),
                Arg::I(ti),
                Arg::I(dii),
                Arg::I(dsi),
                Arg::I(lanes),
            ];
            launch(
                &ctx,
                k.ssm_burnin_nosave_typed.get(w).get(dt),
                seq_grid(),
                &args,
            );
            bytes_of(
                &ctx,
                &[(h.cached_ptr(), st * 4), (y.cached_ptr(), bt * di * es)],
            )
        };
        assert!(
            nosave(IndexWidth::I32) == nosave(IndexWidth::I64),
            "{dt:?} forward without saves"
        );

        // The decode step: B and C from x_proj rows past dt_rank columns,
        // the gate from in_proj rows.
        let (dt_rank, xs) = (5usize, 5 + 2 * ds);
        let xdbl = typed(&ctx, &det(b * xs, 9, 0.6), dt);
        let step = |w: IndexWidth| -> Vec<Vec<u8>> {
            let h = floats(&ctx, &h0);
            let y = DtypedBuf::zeros(&ctx.stream, b * di, dt).expect("y");
            let args = [
                Arg::P(h.cached_ptr()),
                Arg::P(y.cached_ptr()),
                Arg::P(x.cached_ptr()),
                Arg::P(u.cached_ptr()),
                Arg::P(xdbl.cached_ptr()),
                Arg::P(gate.cached_ptr()),
                Arg::I(2 * dii),
                Arg::P(an.cached_ptr()),
                Arg::P(d.cached_ptr()),
                Arg::I(bi),
                Arg::I(dii),
                Arg::I(dsi),
                Arg::I(xs as i32),
                Arg::I(dt_rank as i32),
                Arg::I((dt_rank + ds) as i32),
            ];
            launch(
                &ctx,
                k.ssm_step_fwd_fused_typed.get(w).get(dt),
                grid(),
                &args,
            );
            bytes_of(
                &ctx,
                &[(h.cached_ptr(), st * 4), (y.cached_ptr(), b * di * es)],
            )
        };
        assert!(
            step(IndexWidth::I32) == step(IndexWidth::I64),
            "{dt:?} decode step"
        );

        // The sequential backward over a tape [b, t + 1, d_inner, d_state].
        let tape = floats(&ctx, &det(b * (t + 1) * di * ds, 10, 0.5));
        let dy = typed(&ctx, &det(bt * di, 11, 0.2), dt);
        let backward = |f: &CudaFunction| -> Vec<Vec<u8>> {
            let d_delta = DtypedBuf::zeros(&ctx.stream, bt * di, dt).expect("d_delta");
            let d_u = DtypedBuf::zeros(&ctx.stream, bt * di, dt).expect("d_u");
            let d_b = DtypedBuf::zeros(&ctx.stream, bt * di * ds, dt).expect("d_b");
            let d_c = DtypedBuf::zeros(&ctx.stream, bt * di * ds, dt).expect("d_c");
            let d_d = GpuBuffer::zeros(&ctx.stream, b * di).expect("d_d");
            let d_a = GpuBuffer::zeros(&ctx.stream, st).expect("d_a");
            let args = [
                Arg::P(tape.cached_ptr()),
                Arg::P(x.cached_ptr()),
                Arg::P(u.cached_ptr()),
                Arg::P(bm.cached_ptr()),
                Arg::P(cm.cached_ptr()),
                Arg::P(an.cached_ptr()),
                Arg::P(d.cached_ptr()),
                Arg::P(dy.cached_ptr()),
                Arg::P(d_delta.cached_ptr()),
                Arg::P(d_u.cached_ptr()),
                Arg::P(d_b.cached_ptr()),
                Arg::P(d_c.cached_ptr()),
                Arg::P(d_d.cached_ptr()),
                Arg::P(d_a.cached_ptr()),
                Arg::I(bi),
                Arg::I(ti),
                Arg::I(dii),
                Arg::I(dsi),
                Arg::I(lanes),
            ];
            launch(&ctx, f, seq_grid(), &args);
            bytes_of(
                &ctx,
                &[
                    (d_delta.cached_ptr(), bt * di * es),
                    (d_u.cached_ptr(), bt * di * es),
                    (d_b.cached_ptr(), bt * di * ds * es),
                    (d_c.cached_ptr(), bt * di * ds * es),
                    (d_d.cached_ptr(), b * di * 4),
                    (d_a.cached_ptr(), st * 4),
                ],
            )
        };
        let typed_bwd = &k.ssm_backward_local_typed;
        assert!(
            backward(typed_bwd.i32.get(dt)) == backward(typed_bwd.i64.get(dt)),
            "{dt:?} sequential backward"
        );
        if dt == WeightDtype::F32 {
            let f32_bwd = &k.ssm_backward_local;
            assert!(
                backward(&f32_bwd.i32) == backward(&f32_bwd.i64),
                "the f32 route's sequential backward"
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Bwd {
    b: usize,
    t: usize,
    di: usize,
    ds: usize,
    fold: bool,
    slim: bool,
}

impl Bwd {
    fn tape_len(self) -> usize {
        let n_chunks = self.t.div_ceil(SCAN_CHUNK);
        if self.slim {
            self.b * self.di * self.ds * 3 * n_chunks
        } else {
            self.b * self.di * self.ds * (self.t + 1)
        }
    }

    /// Rows of B and C gradient, one per channel or one per channel group.
    fn grad_rows(self) -> usize {
        if self.fold {
            self.di / SCAN_BWD_DGROUP
        } else {
            self.di
        }
    }

    /// Elements of the per-sample d_a_log rows: one row per chunk in the
    /// fold, one accumulated row otherwise.
    fn a_log_len(self) -> usize {
        if self.fold {
            self.b * self.t.div_ceil(SCAN_CHUNK).max(1) * self.di * self.ds
        } else {
            self.b * self.di * self.ds
        }
    }

    fn cfg(self, dt: WeightDtype) -> LaunchConfig {
        if self.fold {
            grid_parallel_scan_bwd_fold(self.b, self.di, self.ds, dt.size_bytes())
        } else {
            grid_parallel_scan_bwd(self.b, self.di)
        }
    }
}

/// The inputs of a parallel backward launch.
struct BwdIn {
    tape: u64,
    delta: u64,
    raw: u64,
    u: u64,
    bm: u64,
    cm: u64,
    a_neg: u64,
    d: u64,
    dy: u64,
}

/// Every output of one parallel backward launch, as bytes.
fn run_bwd(ctx: &GpuCtx, f: &CudaFunction, s: Bwd, dt: WeightDtype, i: &BwdIn) -> Vec<Vec<u8>> {
    let es = dt.size_bytes();
    let act = s.b * s.t * s.di;
    let grads = s.b * s.ds * s.grad_rows() * s.t;
    let d_delta = DtypedBuf::zeros(&ctx.stream, act, dt).expect("d_delta");
    let d_u = DtypedBuf::zeros(&ctx.stream, act, dt).expect("d_u");
    let d_b = DtypedBuf::zeros(&ctx.stream, grads, dt).expect("d_b");
    let d_c = DtypedBuf::zeros(&ctx.stream, grads, dt).expect("d_c");
    let d_d = GpuBuffer::zeros(&ctx.stream, s.b * s.di).expect("d_d");
    let d_a = GpuBuffer::zeros(&ctx.stream, s.a_log_len()).expect("d_a");
    let mut args = vec![
        Arg::P(i.tape),
        Arg::P(i.delta),
        Arg::P(i.u),
        Arg::P(i.bm),
        Arg::P(i.cm),
        Arg::P(i.a_neg),
        Arg::P(i.d),
        Arg::P(i.dy),
        Arg::P(d_delta.cached_ptr()),
    ];
    if s.fold {
        args.push(Arg::P(i.raw));
    }
    args.extend([
        Arg::P(d_u.cached_ptr()),
        Arg::P(d_b.cached_ptr()),
        Arg::P(d_c.cached_ptr()),
        Arg::P(d_d.cached_ptr()),
        Arg::P(d_a.cached_ptr()),
        Arg::I(s.b as i32),
        Arg::I(s.t as i32),
        Arg::I(s.di as i32),
        Arg::I(s.ds as i32),
        Arg::P(i.tape),
        Arg::I(i32::from(s.slim)),
    ]);
    // Every fold build takes the global carries' scratch last.
    let carry = GpuBuffer::zeros(&ctx.stream, scan_bwd_fold_carry_len(s.b, s.di, s.ds))
        .expect("fold carries");
    if s.fold {
        launch_runs_tail(ctx, f, s.cfg(dt), &args, &[Arg::P(carry.cached_ptr())]);
    } else {
        launch_runs(ctx, f, s.cfg(dt), &args);
    }
    bytes_of(
        ctx,
        &[
            (d_delta.cached_ptr(), act * es),
            (d_u.cached_ptr(), act * es),
            (d_b.cached_ptr(), grads * es),
            (d_c.cached_ptr(), grads * es),
            (d_d.cached_ptr(), s.b * s.di * 4),
            (d_a.cached_ptr(), s.a_log_len() * 4),
        ],
    )
}

/// The parallel backward in both offset widths: the fold and the
/// ungrouped kernel, over the slim and the full tape, two chunks of which
/// the second is partial.
#[test]
fn both_builds_of_the_parallel_backward_give_the_same_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let k = &ctx.kernels;
    for dt in DTYPES {
        for fold in [true, false] {
            for slim in [true, false] {
                let s = Bwd {
                    b: 2,
                    t: 1300,
                    di: 16,
                    ds: 16,
                    fold,
                    slim,
                };
                let act = s.b * s.t * s.di;
                let bc = s.b * s.ds * s.t;
                let tape = floats(&ctx, &det(s.tape_len(), 20, 0.5));
                let delta: Vec<f32> = det(act, 21, 0.3).iter().map(|v| v.abs() + 0.01).collect();
                let delta = typed(&ctx, &delta, dt);
                let raw = typed(&ctx, &det(act, 22, 2.0), dt);
                let u = typed(&ctx, &det(act, 23, 1.0), dt);
                let bm = typed(&ctx, &det(bc, 24, 0.7), dt);
                let cm = typed(&ctx, &det(bc, 25, 0.7), dt);
                let an = floats(&ctx, &a_neg(s.di * s.ds, 26));
                let d = floats(&ctx, &det(s.di, 27, 1.0));
                let dy = typed(&ctx, &det(act, 28, 0.2), dt);
                let i = BwdIn {
                    tape: tape.cached_ptr(),
                    delta: delta.cached_ptr(),
                    raw: raw.cached_ptr(),
                    u: u.cached_ptr(),
                    bm: bm.cached_ptr(),
                    cm: cm.cached_ptr(),
                    a_neg: an.cached_ptr(),
                    d: d.cached_ptr(),
                    dy: dy.cached_ptr(),
                };
                let kernels = if fold {
                    &k.ssm_parallel_bwd_fold_typed
                } else {
                    &k.ssm_parallel_bwd_typed
                };
                let [narrow, wide] =
                    WIDTHS.map(|w| run_bwd(&ctx, kernels.get(w).get(dt), s, dt, &i));
                assert!(narrow == wide, "{dt:?} {s:?}: the builds differ");
            }
        }
    }
}

/// 8193 sequences of 1024 steps over four channels and 256 states: the B
/// and C rows and the fold's B and C gradient rows of the last sequence
/// start at element 2^31, so the int build would wrap every offset into
/// them. The fold takes its `_i64` build, and the first and the last
/// sequence come out exactly as each one does in a launch of its own on
/// the int build.
#[test]
#[ignore = "allocates 18 GiB of device memory"]
fn the_fold_i64_build_gives_each_sequence_of_a_batch_past_i32_max_its_own_bits() {
    let s = Bwd {
        b: 8193,
        t: 1024,
        di: 4,
        ds: 256,
        fold: true,
        slim: true,
    };
    let one = Bwd { b: 1, ..s };
    assert_eq!(
        scan_bwd_index_width(s.b, s.t, s.di, s.ds, true, true),
        IndexWidth::I64
    );
    assert_eq!(
        scan_bwd_index_width(1, s.t, s.di, s.ds, true, true),
        IndexWidth::I32
    );
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let dt = WeightDtype::Bf16;
    let es = dt.size_bytes();
    let kernels = &ctx.kernels.ssm_parallel_bwd_fold_typed;
    let seqs = [0, s.b - 1];
    let (act, slab, tape_n) = (s.t * s.di, s.ds * s.t, s.tape_len() / s.b);

    // B and C hold 0.25 everywhere except the rows of the checked
    // sequences; the per-sequence rows of every other input come from a
    // fixed pattern, copied out for the launches of one sequence.
    let fill = |buf: &DtypedBuf, n: usize| {
        let r = unsafe {
            cudarc::driver::sys::cuMemsetD16Async(buf.cached_ptr(), 0x3e80, n, st.cu_stream())
        };
        assert_eq!(r, CUDA_SUCCESS, "memset");
    };
    let bm = DtypedBuf::zeros(st, s.b * slab, dt).expect("4 GiB of bf16");
    let cm = DtypedBuf::zeros(st, s.b * slab, dt).expect("4 GiB of bf16");
    fill(&bm, s.b * slab);
    fill(&cm, s.b * slab);
    let own_bc: Vec<(DtypedBuf, DtypedBuf)> = seqs
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            let b1 = typed(&ctx, &det(slab, 40 + 2 * i as u32, 0.7), dt);
            let c1 = typed(&ctx, &det(slab, 41 + 2 * i as u32, 0.7), dt);
            let at = (q * slab * es) as u64;
            copy(st, bm.cached_ptr() + at, b1.cached_ptr(), slab * es);
            copy(st, cm.cached_ptr() + at, c1.cached_ptr(), slab * es);
            (b1, c1)
        })
        .collect();
    let delta: Vec<f32> = det(s.b * act, 50, 0.3)
        .iter()
        .map(|v| v.abs() + 0.01)
        .collect();
    let delta = typed(&ctx, &delta, dt);
    let raw = typed(&ctx, &det(s.b * act, 51, 2.0), dt);
    let u = typed(&ctx, &det(s.b * act, 52, 1.0), dt);
    let dy = typed(&ctx, &det(s.b * act, 53, 0.2), dt);
    let tape = floats(&ctx, &det(s.tape_len(), 54, 0.5));
    let an = floats(&ctx, &a_neg(s.di * s.ds, 55));
    let d = floats(&ctx, &det(s.di, 56, 1.0));
    let batch_in = BwdIn {
        tape: tape.cached_ptr(),
        delta: delta.cached_ptr(),
        raw: raw.cached_ptr(),
        u: u.cached_ptr(),
        bm: bm.cached_ptr(),
        cm: cm.cached_ptr(),
        a_neg: an.cached_ptr(),
        d: d.cached_ptr(),
        dy: dy.cached_ptr(),
    };
    let batch = run_bwd(&ctx, kernels.i64.get(dt), s, dt, &batch_in);
    drop((bm, cm));
    let grads = one.ds * one.grad_rows() * one.t;
    let a_log = one.a_log_len();
    for (&q, (b1, c1)) in seqs.iter().zip(&own_bc) {
        let rows_of = |src: &DtypedBuf, per: usize| -> DtypedBuf {
            let dst = DtypedBuf::zeros(st, per, dt).expect("rows");
            copy(
                st,
                dst.cached_ptr(),
                src.cached_ptr() + (q * per * es) as u64,
                per * es,
            );
            dst
        };
        let (delta1, raw1, u1, dy1) = (
            rows_of(&delta, act),
            rows_of(&raw, act),
            rows_of(&u, act),
            rows_of(&dy, act),
        );
        let tape1 = GpuBuffer::zeros(st, tape_n).expect("tape");
        copy(
            st,
            tape1.cached_ptr(),
            tape.cached_ptr() + (q * tape_n * 4) as u64,
            tape_n * 4,
        );
        let own_in = BwdIn {
            tape: tape1.cached_ptr(),
            delta: delta1.cached_ptr(),
            raw: raw1.cached_ptr(),
            u: u1.cached_ptr(),
            bm: b1.cached_ptr(),
            cm: c1.cached_ptr(),
            a_neg: an.cached_ptr(),
            d: d.cached_ptr(),
            dy: dy1.cached_ptr(),
        };
        let own = run_bwd(&ctx, kernels.i32.get(dt), one, dt, &own_in);
        let slices = [
            ("d_delta", act * es),
            ("d_u", act * es),
            ("d_B", grads * es),
            ("d_C", grads * es),
            ("d_D", one.di * 4),
            ("d_a_log", a_log * 4),
        ];
        for (j, (what, per)) in slices.iter().enumerate() {
            assert!(
                batch[j][q * per..(q + 1) * per] == own[j][..],
                "fold past 2^31: {what} of sequence {q} differs from its own launch"
            );
        }
    }
}
