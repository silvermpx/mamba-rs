//! The f32 fold backward with its carries in global memory against the
//! fold that keeps them in shared memory. On the same inputs and tapes
//! every output of the two (the pre-softplus dt and u gradients, the dB/dC
//! group rows, the D and A_log partials) is the same bit for bit, at states
//! on both sides of where a part with 100 KB of shared memory a SM stops
//! fitting two f32 blocks (146), with both index widths and both tapes;
//! the launch the backward takes, in f32, bf16 and f16, gives the shared
//! fold's bits and carries the shared memory of the build it picks at every
//! state, the half folds keeping their carries shared. Ignored: the two f32
//! builds timed, for the A/B on a quiet card.
//!
//!   cargo test --release --features cuda --test m1_fold_carries -- --nocapture
//!   cargo test --release --features cuda --test m1_fold_carries -- --ignored --nocapture

#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use std::time::Instant;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::kernels::MambaKernels;
use mamba_rs::gpu::launch::{
    FoldCarries, IndexWidth, SCAN_BWD_DGROUP, SCAN_CHUNK, SCAN_MAX_DSTATE, grid_parallel_scan,
    grid_parallel_scan_bwd_fold_carries, scan_bwd_fold_carry_len, scan_bwd_fold_smem_bytes,
    scan_tape_len, y_runs,
};

/// What the outputs of the global-carry run start as before the kernel
/// writes them, a quiet NaN the shared-carry run's zeros never equal: a
/// slot the global-carry build leaves unwritten fails the comparison.
const UNWRITTEN: f32 = f32::NAN;

const DTYPES: [WeightDtype; 3] = [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16];

fn make_ctx() -> (GpuCtx, MambaKernels) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let kernels = MambaKernels::compile(ctx.stream.context(), arch::arch0()).unwrap();
    (ctx, kernels)
}

fn det_rand(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFFFF) as f32 / 65536.0 - 0.5
        })
        .collect()
}

fn floats(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut buf = GpuBuffer::zeros(&ctx.stream, data.len()).unwrap();
    buf.upload(&ctx.stream, data).unwrap();
    buf
}

fn typed(ctx: &GpuCtx, data: &[f32], dtype: WeightDtype) -> DtypedBuf {
    let buf = DtypedBuf::zeros(&ctx.stream, data.len(), dtype).unwrap();
    buf.upload_f32(&ctx.stream, data).unwrap();
    buf
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    b: usize,
    t: usize,
    di: usize,
    ds: usize,
}

impl Shape {
    fn rows(self) -> usize {
        self.b * self.t * self.di
    }

    fn chunks(self) -> usize {
        self.t.div_ceil(SCAN_CHUNK).max(1)
    }

    fn tape_len(self, slim: bool) -> usize {
        if slim {
            scan_tape_len(self.b, self.t, self.di, self.ds)
        } else {
            self.b * self.di * self.ds * (self.t + 1)
        }
    }
}

/// One window's inputs in one dtype, with the tape and the dt save its
/// forward wrote.
struct Window {
    shape: Shape,
    dtype: WeightDtype,
    slim: bool,
    delta_raw: DtypedBuf,
    delta: DtypedBuf,
    u: DtypedBuf,
    b_rows: DtypedBuf,
    c_rows: DtypedBuf,
    a_neg: GpuBuffer,
    d: GpuBuffer,
    dy: DtypedBuf,
    tape: GpuBuffer,
}

impl Window {
    fn new(
        ctx: &GpuCtx,
        k: &MambaKernels,
        (shape, dtype): (Shape, WeightDtype),
        slim: bool,
    ) -> Self {
        let Shape { b, t, di, ds } = shape;
        let seed = (b * 7 + t * 13 + di * 17 + ds * 19) as u32 | 1;
        let delta_raw: Vec<f32> = det_rand(shape.rows(), seed ^ 0xA2)
            .into_iter()
            .map(|x| -2.0 + 1.5 * (x + 0.5))
            .collect();
        let window = Self {
            shape,
            dtype,
            slim,
            delta_raw: typed(ctx, &delta_raw, dtype),
            delta: DtypedBuf::zeros(&ctx.stream, shape.rows(), dtype).unwrap(),
            u: typed(ctx, &det_rand(shape.rows(), seed ^ 0xA3), dtype),
            b_rows: typed(ctx, &det_rand(b * ds * t, seed ^ 0xA4), dtype),
            c_rows: typed(ctx, &det_rand(b * ds * t, seed ^ 0xA5), dtype),
            a_neg: floats(
                ctx,
                &(0..di * ds)
                    .map(|i| -0.05 - 0.9 * ((i * 37 % 101) as f32 / 101.0))
                    .collect::<Vec<_>>(),
            ),
            d: floats(
                ctx,
                &(0..di).map(|i| 0.5 + 0.05 * (i as f32)).collect::<Vec<_>>(),
            ),
            dy: typed(ctx, &det_rand(shape.rows(), seed ^ 0xA6), dtype),
            tape: GpuBuffer::zeros(&ctx.stream, shape.tape_len(slim)).unwrap(),
        };
        window.forward(ctx, k);
        window
    }

    /// The forward writes the tape (slim or full; the slim one rides the
    /// same buffer as in training) and the softplus dt the backward reads.
    fn forward(&self, ctx: &GpuCtx, k: &MambaKernels) {
        let Shape { b, t, di, ds } = self.shape;
        let h = GpuBuffer::zeros(&ctx.stream, b * di * ds).unwrap();
        let y = DtypedBuf::zeros(&ctx.stream, self.shape.rows(), self.dtype).unwrap();
        let ptrs = [
            h.cached_ptr(),
            y.cached_ptr(),
            self.tape.cached_ptr(),
            self.delta_raw.cached_ptr(),
            self.delta.cached_ptr(),
            self.u.cached_ptr(),
            self.b_rows.cached_ptr(),
            self.c_rows.cached_ptr(),
            self.a_neg.cached_ptr(),
            self.d.cached_ptr(),
        ];
        let dims = [b as i32, t as i32, di as i32, ds as i32];
        let tape = self.tape.cached_ptr();
        let slim = i32::from(self.slim);
        for (y_first, cfg) in y_runs(grid_parallel_scan(b, t, di, ds)) {
            let mut bld = ctx
                .stream
                .launch_builder(k.ssm_parallel_fwd_typed.i32.get(self.dtype));
            for p in &ptrs {
                bld.arg(p);
            }
            for x in &dims {
                bld.arg(x);
            }
            bld.arg(&tape);
            bld.arg(&slim);
            bld.arg(&y_first);
            unsafe { bld.launch(cfg) }.unwrap();
        }
        ctx.stream.synchronize().unwrap();
    }
}

/// A fold backward's outputs and its global carries' scratch.
struct Grads {
    d_delta_raw: DtypedBuf,
    d_u: DtypedBuf,
    d_b: DtypedBuf,
    d_c: DtypedBuf,
    d_d: GpuBuffer,
    d_a_log: GpuBuffer,
    carry: GpuBuffer,
}

impl Grads {
    fn filled(ctx: &GpuCtx, (s, dtype): (Shape, WeightDtype), fill: f32) -> Self {
        let bc = s.b * s.t * (s.di / SCAN_BWD_DGROUP) * s.ds;
        let typed_fill = |n: usize| typed(ctx, &vec![fill; n], dtype);
        Self {
            d_delta_raw: typed_fill(s.rows()),
            d_u: typed_fill(s.rows()),
            d_b: typed_fill(bc),
            d_c: typed_fill(bc),
            d_d: floats(ctx, &vec![fill; s.b * s.di]),
            d_a_log: floats(ctx, &vec![fill; s.b * s.chunks() * s.di * s.ds]),
            carry: GpuBuffer::zeros(&ctx.stream, scan_bwd_fold_carry_len(s.b, s.di, s.ds)).unwrap(),
        }
    }

    fn read(&self, ctx: &GpuCtx) -> [Vec<f32>; 6] {
        let typed_out = |buf: &DtypedBuf| {
            let mut v = vec![0f32; buf.len_elems()];
            buf.download_f32(&ctx.stream, &mut v).unwrap();
            v
        };
        let float_out = |buf: &GpuBuffer, n: usize| {
            let mut v = vec![0f32; n];
            buf.download(&ctx.stream, &mut v).unwrap();
            v
        };
        let out = [
            typed_out(&self.d_delta_raw),
            typed_out(&self.d_u),
            typed_out(&self.d_b),
            typed_out(&self.d_c),
            float_out(&self.d_d, self.d_d.len()),
            float_out(&self.d_a_log, self.d_a_log.len()),
        ];
        ctx.stream.synchronize().unwrap();
        out
    }
}

const OUTPUTS: [&str; 6] = ["d_delta_raw", "d_u", "d_B", "d_C", "d_D", "d_A_log"];

/// One fold backward over the window into `out`, its global carries'
/// scratch last as every fold build takes it.
fn fold(ctx: &GpuCtx, (kernel, cfg): (&CudaFunction, LaunchConfig), w: &Window, out: &Grads) {
    let Shape { b, t, di, ds } = w.shape;
    let ptrs = [
        w.tape.cached_ptr(),
        w.delta.cached_ptr(),
        w.u.cached_ptr(),
        w.b_rows.cached_ptr(),
        w.c_rows.cached_ptr(),
        w.a_neg.cached_ptr(),
        w.d.cached_ptr(),
        w.dy.cached_ptr(),
        out.d_delta_raw.cached_ptr(),
        w.delta_raw.cached_ptr(),
        out.d_u.cached_ptr(),
        out.d_b.cached_ptr(),
        out.d_c.cached_ptr(),
        out.d_d.cached_ptr(),
        out.d_a_log.cached_ptr(),
    ];
    let dims = [b as i32, t as i32, di as i32, ds as i32];
    let tape = w.tape.cached_ptr();
    let slim = i32::from(w.slim);
    let carry = out.carry.cached_ptr();
    for (y_first, cfg) in y_runs(cfg) {
        let mut bld = ctx.stream.launch_builder(kernel);
        for p in &ptrs {
            bld.arg(p);
        }
        for x in &dims {
            bld.arg(x);
        }
        bld.arg(&tape);
        bld.arg(&slim);
        bld.arg(&y_first);
        bld.arg(&carry);
        unsafe { bld.launch(cfg) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();
}

/// The fold of a dtype and index width with its carries where `carries`
/// says, and the launch config of that build.
fn build(
    k: &MambaKernels,
    (s, dtype): (Shape, WeightDtype),
    width: IndexWidth,
    carries: FoldCarries,
) -> (&CudaFunction, LaunchConfig) {
    let kernel = match carries {
        FoldCarries::Shared => k.ssm_parallel_bwd_fold_typed.get(width).get(dtype),
        FoldCarries::Global => {
            assert_eq!(
                dtype,
                WeightDtype::F32,
                "only the f32 fold has a global-carry build"
            );
            k.ssm_parallel_bwd_fold_carry_f32.get(width)
        }
    };
    (
        kernel,
        grid_parallel_scan_bwd_fold_carries(s.b, s.di, s.ds, dtype.size_bytes(), carries),
    )
}

/// The global-carry run's outputs against the shared-carry run's, bit for
/// bit; returns the slots compared.
fn same_bits(label: &str, shared: &[Vec<f32>; 6], global: &[Vec<f32>; 6]) -> usize {
    let mut compared = 0;
    for ((name, a), b) in OUTPUTS.iter().zip(shared).zip(global) {
        assert_eq!(a.len(), b.len());
        if let Some((i, (x, y))) = a
            .iter()
            .zip(b)
            .enumerate()
            .find(|(_, (x, y))| x.to_bits() != y.to_bits())
        {
            panic!(
                "{label} {name}[{i}]: shared carries {x:e} ({:#010x}), global {y:e} ({:#010x})",
                x.to_bits(),
                y.to_bits()
            );
        }
        assert!(
            a.iter().all(|x| x.is_finite()),
            "{label} {name}: a non-finite output"
        );
        compared += a.len();
    }
    compared
}

#[test]
fn the_global_carry_fold_keeps_the_shared_fold_bits() {
    let (ctx, k) = make_ctx();
    // States on both sides of 146, up to the scan's largest;
    // windows of one partial chunk, whole chunks and several chunks with a
    // partial last.
    let shapes = [
        Shape {
            b: 3,
            t: 1500,
            di: 32,
            ds: 16,
        },
        Shape {
            b: 2,
            t: 700,
            di: 64,
            ds: 87,
        },
        Shape {
            b: 1,
            t: 2500,
            di: 16,
            ds: 88,
        },
        Shape {
            b: 2,
            t: 700,
            di: 64,
            ds: 146,
        },
        Shape {
            b: 2,
            t: 1100,
            di: 16,
            ds: 147,
        },
        Shape {
            b: 1,
            t: 1024,
            di: 64,
            ds: 200,
        },
        Shape {
            b: 2,
            t: 2500,
            di: 8,
            ds: 256,
        },
    ];
    for shape in shapes {
        for dtype in DTYPES {
            for slim in [true, false] {
                let w = Window::new(&ctx, &k, (shape, dtype), slim);
                // Only the f32 fold has a global-carry build.
                let widths: &[IndexWidth] = if dtype == WeightDtype::F32 {
                    &[IndexWidth::I32, IndexWidth::I64]
                } else {
                    &[]
                };
                for &width in widths {
                    let a = Grads::filled(&ctx, (shape, dtype), 0.0);
                    fold(
                        &ctx,
                        build(&k, (shape, dtype), width, FoldCarries::Shared),
                        &w,
                        &a,
                    );
                    let b = Grads::filled(&ctx, (shape, dtype), UNWRITTEN);
                    fold(
                        &ctx,
                        build(&k, (shape, dtype), width, FoldCarries::Global),
                        &w,
                        &b,
                    );
                    let label = format!("{shape:?} {dtype:?} slim={slim} {width:?}");
                    let n = same_bits(&label, &a.read(&ctx), &b.read(&ctx));
                    eprintln!("  {label}: {n} outputs the same bits");
                }
                // The launch the backward takes, whichever build it picks.
                let a = Grads::filled(&ctx, (shape, dtype), 0.0);
                fold(
                    &ctx,
                    build(&k, (shape, dtype), IndexWidth::I32, FoldCarries::Shared),
                    &w,
                    &a,
                );
                let b = Grads::filled(&ctx, (shape, dtype), UNWRITTEN);
                fold(
                    &ctx,
                    k.ssm_parallel_bwd_fold_launch(dtype, shape.b, shape.t, shape.di, shape.ds),
                    &w,
                    &b,
                );
                same_bits(
                    &format!("{shape:?} {dtype:?} slim={slim} the backward's launch"),
                    &a.read(&ctx),
                    &b.read(&ctx),
                );
            }
        }
    }
}

#[test]
fn the_launch_carries_the_shared_memory_of_its_build() {
    let (_ctx, k) = make_ctx();
    for dtype in DTYPES {
        let mut global = Vec::new();
        for ds in 1..=SCAN_MAX_DSTATE {
            let (_, cfg) = k.ssm_parallel_bwd_fold_launch(dtype, 2, 2048, 64, ds);
            let carries = k.fold_carries(dtype, IndexWidth::I32, ds);
            if carries == FoldCarries::Global {
                assert_eq!(dtype, WeightDtype::F32, "{dtype:?} at {ds} states");
                global.push(ds);
            }
            assert_eq!(
                cfg.shared_mem_bytes as usize,
                scan_bwd_fold_smem_bytes(ds, dtype.size_bytes(), carries),
                "{dtype:?} at {ds} states"
            );
        }
        eprintln!(
            "  the {dtype:?} fold keeps its carries in global memory on this device at {} of \
             {SCAN_MAX_DSTATE} states: {:?}",
            global.len(),
            global.first().zip(global.last())
        );
    }
}

/// Median microseconds of a fold launch over the window.
fn timed(ctx: &GpuCtx, launch: (&CudaFunction, LaunchConfig), w: &Window, out: &Grads) -> f64 {
    for _ in 0..3 {
        fold(ctx, launch, w, out);
    }
    let mut samples: Vec<f64> = (0..9)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..5 {
                fold(ctx, launch, w, out);
            }
            start.elapsed().as_secs_f64() * 1e6 / 5.0
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[4]
}

#[test]
#[ignore = "needs a quiet card - measurement"]
fn the_global_carry_fold_timed() {
    let (ctx, k) = make_ctx();
    // The dB and dC group rows take b * t * di / 4 * ds elements each;
    // FOLD_B, FOLD_T and FOLD_DI shrink them on a card with less free
    // memory.
    let dim = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default)
    };
    let (b, t, di) = (dim("FOLD_B", 1), dim("FOLD_T", 1100), dim("FOLD_DI", 1024));
    for dtype in [WeightDtype::F32] {
        for ds in [64, 128, 146, 147, 160, 192, 224, 256] {
            let shape = Shape { b, t, di, ds };
            let w = Window::new(&ctx, &k, (shape, dtype), true);
            let out = Grads::filled(&ctx, (shape, dtype), 0.0);
            let shared = build(&k, (shape, dtype), IndexWidth::I32, FoldCarries::Shared);
            let global = build(&k, (shape, dtype), IndexWidth::I32, FoldCarries::Global);
            let mut shared_us = Vec::new();
            let mut global_us = Vec::new();
            // Interleaved, so a load on the card falls on both alike.
            for _ in 0..3 {
                shared_us.push(timed(&ctx, shared, &w, &out));
                global_us.push(timed(&ctx, global, &w, &out));
            }
            shared_us.sort_by(f64::total_cmp);
            global_us.sort_by(f64::total_cmp);
            println!(
                "fold {dtype:?} b={b} t={t} di={di} ds={ds}: carries shared {:.1} us, global \
                 {:.1} us ({:.3}x), the launch keeps them {:?}",
                shared_us[1],
                global_us[1],
                shared_us[1] / global_us[1],
                k.fold_carries(dtype, IndexWidth::I32, ds)
            );
        }
    }
}
