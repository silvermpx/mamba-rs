//! The grouped Mamba-1 parallel scan backward against the released kernels.
//! A window of at most 512 steps is walked back by the forward's channel
//! groups, side by side; the released fold kernel walked four channels a
//! block in turn and the ungrouped kernel one, each on all 128 threads. Over
//! the same forward, the grouped kernel's outputs must equal the released
//! kernel's bit for bit: the pre-softplus dt gradient, d_u, the dB and dC
//! rows, d_D and the d_a_log slots. Where four divide d_inner its rows sum
//! four channels and it is held to the fold kernel; elsewhere they sum one
//! and it is held to the ungrouped kernel, whose post-softplus gradient the
//! library's softplus backward takes on. One allowance: the released
//! kernels composed and reduced through threads past the window, which turn
//! a negative zero positive, so a zero may differ in its sign.
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;
#[path = "common/frozen_kernels.rs"]
mod frozen_kernels;

use std::sync::Arc;

use cudarc::driver::sys::CUfunction_attribute_enum as FnAttr;
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{
    SCAN_BWD_DGROUP, grid_1d_wide, grid_parallel_scan, grid_parallel_scan_bwd,
    grid_parallel_scan_bwd_fold, grid_parallel_scan_bwd_groups, scan_tape_len, y_runs,
};

const DTYPES: [WeightDtype; 3] = [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16];

/// Windows on both sides of every group width's span below the block's.
const WINDOWS: [usize; 16] = [
    1, 7, 8, 9, 63, 64, 65, 100, 128, 129, 255, 256, 257, 300, 511, 512,
];

/// `(d_inner, d_state)`: channel counts four divide (rows of four, against
/// the fold kernel), one with fold groups past the last channel, and counts
/// four do not divide (rows of one, against the ungrouped kernel).
const CHANNELS: [(usize, usize); 7] = [
    (4, 16),
    (24, 1),
    (36, 64),
    (32, 256),
    (1, 16),
    (33, 64),
    (7, 256),
];

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

fn suffix(dt: WeightDtype) -> &'static str {
    match dt {
        WeightDtype::F32 => "f32",
        WeightDtype::Bf16 => "bf16",
        _ => "f16",
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    b: usize,
    t: usize,
    di: usize,
    ds: usize,
}

/// One forward's tape and the rows both backwards read.
struct Walked {
    tape: GpuBuffer,
    delta_raw: DtypedBuf,
    delta: DtypedBuf,
    u: DtypedBuf,
    bm: DtypedBuf,
    cm: DtypedBuf,
    a_neg: GpuBuffer,
    d: GpuBuffer,
    dy: DtypedBuf,
}

/// The saving forward over fresh rows: its tape (slim or full, one buffer
/// as the model passes it) and the post-softplus deltas it saved.
fn walk(ctx: &GpuCtx, s: Shape, dt: WeightDtype, slim: bool) -> Walked {
    let st = &ctx.stream;
    let n = s.b * s.t * s.di;
    let a_neg: Vec<f32> = det(s.di * s.ds, 15, 1.0)
        .iter()
        .map(|v| -(v.abs() + 0.1))
        .collect();
    let w = Walked {
        tape: GpuBuffer::zeros(
            st,
            if slim {
                scan_tape_len(s.b, s.t, s.di, s.ds)
            } else {
                s.b * s.di * s.ds * (s.t + 1)
            },
        )
        .expect("tape"),
        delta_raw: typed(ctx, &det(n, 11, 1.5), dt),
        delta: DtypedBuf::zeros(st, n, dt).expect("delta"),
        u: typed(ctx, &det(n, 12, 1.0), dt),
        bm: typed(ctx, &det(s.b * s.ds * s.t, 13, 0.7), dt),
        cm: typed(ctx, &det(s.b * s.ds * s.t, 14, 0.7), dt),
        a_neg: floats(ctx, &a_neg),
        d: floats(ctx, &det(s.di, 16, 1.0)),
        dy: typed(ctx, &det(n, 19, 1.0), dt),
    };
    let h = floats(ctx, &det(s.b * s.di * s.ds, 17, 1.0));
    let y = DtypedBuf::zeros(st, n, dt).expect("y");
    let ints = [s.b, s.t, s.di, s.ds].map(|v| v as i32);
    let slim_i = i32::from(slim);
    let ptrs = [
        h.cached_ptr(),
        y.cached_ptr(),
        w.tape.cached_ptr(),
        w.delta_raw.cached_ptr(),
        w.delta.cached_ptr(),
        w.u.cached_ptr(),
        w.bm.cached_ptr(),
        w.cm.cached_ptr(),
        w.a_neg.cached_ptr(),
        w.d.cached_ptr(),
    ];
    let tape_ptr = w.tape.cached_ptr();
    for (y_first, cfg) in y_runs(grid_parallel_scan(s.b, s.t, s.di, s.ds)) {
        let mut bld = st.launch_builder(ctx.kernels.ssm_parallel_fwd_typed.i32.get(dt));
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
    w
}

/// The outputs of a walk back: the pre-softplus dt gradient, d_u, the dB
/// and dC rows, d_D and the d_a_log slots.
type Back = [Vec<f32>; 6];

/// One walk back of `w` by `f` over `grid`, its dB/dC rows `rows` deep;
/// `fold` is the grouped kernel's last argument, and `inline_softplus`
/// whether the kernel takes the pre-softplus rows and emits the pre-softplus
/// gradient itself (else the library's softplus backward follows).
fn walk_back(
    ctx: &GpuCtx,
    (f, grid): (&CudaFunction, LaunchConfig),
    (s, dt, slim): (Shape, WeightDtype, bool),
    w: &Walked,
    (rows, fold, inline_softplus): (usize, Option<i32>, bool),
) -> Back {
    let st = &ctx.stream;
    let n = s.b * s.t * s.di;
    let local = s.b * s.ds * rows * s.t;
    let d_delta = DtypedBuf::zeros(st, n, dt).expect("d_delta");
    let d_u = DtypedBuf::zeros(st, n, dt).expect("d_u");
    let d_b = DtypedBuf::zeros(st, local, dt).expect("d_b");
    let d_c = DtypedBuf::zeros(st, local, dt).expect("d_c");
    let d_d = GpuBuffer::zeros(st, s.b * s.di).expect("d_d");
    let d_a = GpuBuffer::zeros(st, s.b * s.di * s.ds).expect("d_a");
    let head = [
        w.tape.cached_ptr(),
        w.delta.cached_ptr(),
        w.u.cached_ptr(),
        w.bm.cached_ptr(),
        w.cm.cached_ptr(),
        w.a_neg.cached_ptr(),
        w.d.cached_ptr(),
        w.dy.cached_ptr(),
        d_delta.cached_ptr(),
    ];
    let raw = w.delta_raw.cached_ptr();
    let tail = [
        d_u.cached_ptr(),
        d_b.cached_ptr(),
        d_c.cached_ptr(),
        d_d.cached_ptr(),
        d_a.cached_ptr(),
    ];
    let ints = [s.b, s.t, s.di, s.ds].map(|v| v as i32);
    let tape_ptr = w.tape.cached_ptr();
    let slim_i = i32::from(slim);
    for (y_first, cfg) in y_runs(grid) {
        let mut bld = st.launch_builder(f);
        for p in &head {
            bld.arg(p);
        }
        if inline_softplus {
            bld.arg(&raw);
        }
        for p in &tail {
            bld.arg(p);
        }
        for v in &ints {
            bld.arg(v);
        }
        bld.arg(&tape_ptr);
        bld.arg(&slim_i);
        bld.arg(&y_first);
        if let Some(fold) = &fold {
            bld.arg(fold);
        }
        unsafe { bld.launch(cfg) }.expect("walk back");
    }
    let d_raw = if inline_softplus {
        d_delta
    } else {
        let out = DtypedBuf::zeros(st, n, dt).expect("d_raw");
        let (dx, xs, dy, count) = (
            out.cached_ptr(),
            w.delta_raw.cached_ptr(),
            d_delta.cached_ptr(),
            n as i64,
        );
        let mut bld = st.launch_builder(ctx.kernels.softplus_bwd_typed.get(dt));
        bld.arg(&dx);
        bld.arg(&xs);
        bld.arg(&dy);
        bld.arg(&count);
        unsafe { bld.launch(grid_1d_wide(n)) }.expect("softplus backward");
        out
    };
    st.synchronize().expect("sync");
    [
        typed_values(ctx, &d_raw, n),
        typed_values(ctx, &d_u, n),
        typed_values(ctx, &d_b, local),
        typed_values(ctx, &d_c, local),
        d_d.to_cpu(st).expect("d_d"),
        d_a.to_cpu(st).expect("d_a"),
    ]
}

/// A released kernel's entry for `dt`, let take the shared memory `cfg`
/// asks for.
fn released(
    frozen: &Arc<CudaModule>,
    stem: &str,
    dt: WeightDtype,
    cfg: LaunchConfig,
) -> CudaFunction {
    let f = frozen
        .load_function(&format!("{stem}_{}", suffix(dt)))
        .expect("released kernel");
    f.set_attribute(
        FnAttr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        cfg.shared_mem_bytes as i32,
    )
    .expect("shared memory");
    f
}

#[test]
fn m1_grouped_scan_backward_keeps_the_bits_of_the_released_kernels() {
    let device = GpuDevice::new(0).expect("device");
    let ctx = GpuCtx::new(&device).expect("context");
    let frozen = frozen_kernels::frozen_module(
        device.context(),
        arch::arch0(),
        None,
        include_str!("fixtures/m1_parallel_bwd_reference.cu"),
    );
    let mut cases = 0;
    for t in WINDOWS {
        for (di, ds) in CHANNELS {
            let s = Shape { b: 2, t, di, ds };
            let fold = if di % SCAN_BWD_DGROUP == 0 {
                SCAN_BWD_DGROUP
            } else {
                1
            };
            let rows = di / fold;
            for dt in DTYPES {
                for slim in [true, false] {
                    let w = walk(&ctx, s, dt, slim);
                    let old = if fold == SCAN_BWD_DGROUP {
                        let cfg = grid_parallel_scan_bwd_fold(s.b, di, ds, dt.size_bytes());
                        let f = released(&frozen, "ssm_parallel_scan_bwd_fold", dt, cfg);
                        walk_back(&ctx, (&f, cfg), (s, dt, slim), &w, (rows, None, true))
                    } else {
                        let cfg = grid_parallel_scan_bwd(s.b, di);
                        let f = released(&frozen, "ssm_parallel_scan_bwd", dt, cfg);
                        walk_back(&ctx, (&f, cfg), (s, dt, slim), &w, (rows, None, false))
                    };
                    let new = walk_back(
                        &ctx,
                        (
                            ctx.kernels.ssm_parallel_bwd_groups_typed.i32.get(dt),
                            grid_parallel_scan_bwd_groups(s.b, t, di, fold),
                        ),
                        (s, dt, slim),
                        &w,
                        (rows, Some(fold as i32), true),
                    );
                    for (what, (o, n)) in
                        ["dt gradient", "d_u", "dB rows", "dC rows", "d_D", "d_a_log"]
                            .iter()
                            .zip(old.iter().zip(&new))
                    {
                        assert_eq!(
                            bits(o),
                            bits(n),
                            "the grouped backward's {what} at {s:?} {dt:?}, slim {slim}, \
                             rows of {fold}"
                        );
                    }
                    cases += 1;
                }
            }
        }
    }
    println!("m1 grouped scan backward: the released kernels' bits on {cases} cases");
}
