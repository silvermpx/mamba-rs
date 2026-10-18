//! Mamba-3 SISO kernels over grid axes past the y extent of 65535 blocks.
//! Such an axis is launched in runs, each run passing its first block. A
//! head in a later run must come out with the bits of its twin, a head of
//! the first run given the same inputs: the sequential burn-in and its
//! backward, the decode step on both head tiers, and the pre-state and the
//! chunk scan on their whole-row and their sliced tiles. Every other split
//! launch of the family is proven at ordinary shapes by the suites run with
//! `MAMBA_RS_GRID_RUN`.
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use std::sync::Arc;

use cudarc::driver::sys::CUresult::CUDA_SUCCESS;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::launch::{GRID_YZ_EXTENT, y_runs};
use mamba_rs::mamba3_siso::gpu::kernels::{
    Mamba3Kernels, chunk_fused_cfg, chunk_pre_state_sliced_cfg, chunk_scan_coop_cfg,
    chunk_scan_sliced_cfg, step_head_cfg,
};
use mamba_rs::mamba3_siso::route::{
    HeadTier, sequential_backward_sliced_smem_bytes, sequential_forward_sliced_smem_bytes,
    sequential_sliced_block_threads,
};

/// Heads `h < PAIRS` and their twins `h + SHIFT`. SHIFT is two full runs of
/// the y axis, so a twin sits in the third run.
const PAIRS: usize = 16;
const SHIFT: usize = 2 * 65536;
const NH: usize = SHIFT + PAIRS;
const CS: usize = 64;

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

/// `n` floats of NaN, so a value no launch wrote cannot pass for one.
fn poisoned(ctx: &GpuCtx, n: usize) -> GpuBuffer {
    let buf = GpuBuffer::zeros(&ctx.stream, n).expect("alloc");
    let r = unsafe {
        cudarc::driver::sys::cuMemsetD32Async(
            buf.cached_ptr(),
            0x7fc0_0001,
            n,
            ctx.stream.cu_stream(),
        )
    };
    assert_eq!(r, CUDA_SUCCESS, "poison");
    buf
}

/// The words of a buffer, read once the stream drains.
fn words(stream: &Arc<CudaStream>, buf: &GpuBuffer) -> Vec<u32> {
    stream.synchronize().expect("sync");
    let n = buf.size_bytes() / 4;
    let mut out = vec![0u32; n];
    let r = unsafe {
        cudarc::driver::sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), buf.cached_ptr(), n * 4)
    };
    assert_eq!(r, CUDA_SUCCESS, "read");
    out
}

/// A kernel argument.
#[derive(Clone, Copy)]
enum Arg {
    P(u64),
    I(i32),
}

fn launch(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    let mut b = ctx.stream.launch_builder(f);
    for a in args {
        match a {
            Arg::P(p) => b.arg(p),
            Arg::I(i) => b.arg(i),
        };
    }
    unsafe { b.launch(cfg) }.expect("launch");
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

/// [`launch_runs`] for the kernels that take a packing after the run's
/// first block (none here).
fn launch_runs_packed(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    for (y_first, cfg) in y_runs(cfg) {
        let mut run = args.to_vec();
        run.extend([Arg::I(y_first), Arg::P(0)]);
        launch(ctx, f, cfg, &run);
    }
}

fn compile(ctx: &GpuCtx) -> Mamba3Kernels {
    Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), 64)
        .expect("M3 kernels")
}

/// A head-major layout: `outer` blocks of `NH` heads of `row` values.
#[derive(Clone, Copy)]
struct Heads {
    outer: usize,
    row: usize,
}

impl Heads {
    fn len(self) -> usize {
        self.outer * NH * self.row
    }

    /// Copy every head `h < PAIRS` onto its twin.
    fn pair_up(self, v: &mut [f32]) {
        for o in 0..self.outer {
            for h in 0..PAIRS {
                let from = (o * NH + h) * self.row;
                v.copy_within(from..from + self.row, from + SHIFT * self.row);
            }
        }
    }

    /// Whether every head `h < PAIRS` holds its twin's bits.
    fn twins_match(self, w: &[u32]) -> bool {
        assert_eq!(w.len(), self.len());
        (0..self.outer).all(|o| {
            (0..PAIRS).all(|h| {
                let a = (o * NH + h) * self.row;
                let b = a + SHIFT * self.row;
                w[a..a + self.row] == w[b..b + self.row]
            })
        })
    }

    /// An input of this layout whose twins hold the same values.
    fn input(self, ctx: &GpuCtx, seed: u32, map: impl Fn(f32) -> f32) -> GpuBuffer {
        let mut v: Vec<f32> = det(self.len(), seed, 1.0).into_iter().map(map).collect();
        self.pair_up(&mut v);
        floats(ctx, &v)
    }
}

/// The sequential burn-in with its tape, then its backward, over heads
/// past the y extent on the warp tier and on the sliced tier, one block per
/// (sequence, head): every output of a twin head carries its first-run
/// twin's bits.
#[test]
fn the_sequential_scans_give_heads_past_the_y_extent_their_twins_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let k = compile(&ctx);
    let (t, hd, ds) = (3usize, 4usize, 4usize);
    let rows = Heads { outer: t, row: hd };
    let krows = Heads { outer: t, row: ds };
    let per_step = Heads { outer: t, row: 1 };
    let state = Heads {
        outer: 1,
        row: hd * ds,
    };
    let kstate = Heads { outer: 1, row: ds };
    let vstate = Heads { outer: 1, row: hd };
    let per_head = Heads { outer: 1, row: 1 };
    let warp = LaunchConfig {
        grid_dim: (1, NH as u32, 1),
        block_dim: (hd as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let sliced = |smem: usize| LaunchConfig {
        grid_dim: (1, NH as u32, 1),
        block_dim: (sequential_sliced_block_threads(hd, ds) as u32, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let tiers = [
        (
            "warp",
            (&k.m3_burnin_fwd.i32, warp),
            (&k.m3_backward_seq.i32, warp),
        ),
        (
            "sliced",
            (
                &k.m3_burnin_fwd_sliced.f32,
                sliced(sequential_forward_sliced_smem_bytes(hd, ds)),
            ),
            (
                &k.m3_backward_seq_sliced.f32,
                sliced(sequential_backward_sliced_smem_bytes(hd, ds)),
            ),
        ),
    ];
    for (tier, (fwd, fwd_cfg), (bwd, bwd_cfg)) in tiers {
        let same = |v: f32| 0.5 * v;
        let ssm = state.input(&ctx, 1, same);
        let k_state = kstate.input(&ctx, 2, same);
        let v_state = vstate.input(&ctx, 3, same);
        let x = rows.input(&ctx, 4, same);
        let kk = krows.input(&ctx, 5, same);
        let q = krows.input(&ctx, 6, same);
        // The decays stay in (0, 1) so the walk stays finite.
        let alpha = per_step.input(&ctx, 7, |v| 0.55 + 0.4 * v);
        let beta = per_step.input(&ctx, 8, same);
        let gamma = per_step.input(&ctx, 9, same);
        let d = per_head.input(&ctx, 10, |v| v);
        let dy = rows.input(&ctx, 11, |v| 0.2 * v);
        let y = poisoned(&ctx, rows.len());
        let h_saved = poisoned(&ctx, (t + 1) * NH * hd * ds);
        let k_prev = poisoned(&ctx, krows.len());
        let v_prev = poisoned(&ctx, rows.len());
        let ints = [1, t, NH, hd, ds].map(|v| Arg::I(v as i32));
        let mut args: Vec<Arg> = [
            &ssm, &k_state, &v_state, &y, &h_saved, &k_prev, &v_prev, &x, &kk, &q, &alpha, &beta,
            &gamma, &d,
        ]
        .iter()
        .map(|b| Arg::P(b.cached_ptr()))
        .collect();
        args.extend(ints);
        launch_runs_packed(&ctx, fwd, fwd_cfg, &args);
        let st = &ctx.stream;
        for (what, buf, layout) in [
            ("y", &y, rows),
            ("state", &ssm, state),
            ("k state", &k_state, kstate),
            ("v state", &v_state, vstate),
            ("k prev", &k_prev, krows),
            ("v prev", &v_prev, rows),
        ] {
            assert!(
                layout.twins_match(&words(st, buf)),
                "{tier} burn-in: {what}"
            );
        }

        let dx = poisoned(&ctx, rows.len());
        let dk = poisoned(&ctx, krows.len());
        let dq = poisoned(&ctx, krows.len());
        let dalpha = poisoned(&ctx, per_step.len());
        let dbeta = poisoned(&ctx, per_step.len());
        let dgamma = poisoned(&ctx, per_step.len());
        let dd = poisoned(&ctx, NH * hd);
        let mut args: Vec<Arg> = [
            &h_saved, &k_prev, &v_prev, &x, &kk, &q, &alpha, &beta, &gamma, &d, &dy, &dx, &dk, &dq,
            &dalpha, &dbeta, &dgamma, &dd,
        ]
        .iter()
        .map(|b| Arg::P(b.cached_ptr()))
        .collect();
        args.extend(ints);
        launch_runs_packed(&ctx, bwd, bwd_cfg, &args);
        for (what, buf, layout) in [
            ("dx", &dx, rows),
            ("dk", &dk, krows),
            ("dq", &dq, krows),
            ("d alpha", &dalpha, per_step),
            ("d beta", &dbeta, per_step),
            ("d gamma", &dgamma, per_step),
            ("d D", &dd, vstate),
        ] {
            assert!(
                layout.twins_match(&words(st, buf)),
                "{tier} backward: {what}"
            );
        }
    }
}

/// The decode step over heads past the y extent, on the warp tier and on
/// the block tier: every twin head's output and state carry the bits of
/// its first-run twin.
#[test]
fn the_decode_step_gives_heads_past_the_y_extent_their_twins_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let k = compile(&ctx);
    let (batch, hd, ds) = (2usize, 4usize, 8usize);
    let rows = Heads {
        outer: batch,
        row: hd,
    };
    let krows = Heads {
        outer: batch,
        row: ds,
    };
    let state = Heads {
        outer: batch,
        row: hd * ds,
    };
    let per_seq = Heads {
        outer: batch,
        row: 1,
    };
    let same = |v: f32| 0.5 * v;
    for tier in [HeadTier::Warp, HeadTier::Block] {
        let ssm = state.input(&ctx, 21, same);
        let k_state = krows.input(&ctx, 22, same);
        let v_state = rows.input(&ctx, 23, same);
        let x = rows.input(&ctx, 24, |v| v);
        let kc = krows.input(&ctx, 25, |v| v);
        let qc = krows.input(&ctx, 26, |v| v);
        let dt = per_seq.input(&ctx, 27, |v| 0.01 + 0.3 * v.abs());
        let a = per_seq.input(&ctx, 28, |v| -0.1 - v.abs());
        let trap = per_seq.input(&ctx, 29, |v| 0.5 + 0.4 * v);
        let d = Heads { outer: 1, row: 1 }.input(&ctx, 30, |v| v);
        let y = poisoned(&ctx, rows.len());
        let f = match tier {
            HeadTier::Warp => &k.m3_step_fwd.i32,
            HeadTier::Block => &k.m3_step_fwd_block_typed.i32.f32,
        };
        let cfg = step_head_cfg(tier, batch, NH, hd, ds, k.step_block_threads.i32);
        assert!(cfg.grid_dim.1 > GRID_YZ_EXTENT);
        let mut args: Vec<Arg> = [
            &ssm, &k_state, &v_state, &y, &x, &kc, &qc, &dt, &a, &trap, &d,
        ]
        .iter()
        .map(|b| Arg::P(b.cached_ptr()))
        .collect();
        args.extend([batch, NH, hd, ds].map(|v| Arg::I(v as i32)));
        launch_runs(&ctx, f, cfg, &args);
        let st = &ctx.stream;
        for (what, buf, layout) in [
            ("y", &y, rows),
            ("state", &ssm, state),
            ("k state", &k_state, krows),
            ("v state", &v_state, rows),
        ] {
            assert!(layout.twins_match(&words(st, buf)), "{tier:?} step: {what}");
        }
    }
}

/// The chunk scan over heads past the y extent, on its cooperative tiles of
/// one head per block and on its sliced tiles: every twin head's output
/// carries the bits of its first-run twin.
#[test]
fn the_chunk_scan_gives_heads_past_the_y_extent_their_twins_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let k = compile(&ctx);
    let (t, hd, ds) = (CS, 4usize, 4usize);
    let nc = t.div_ceil(CS);
    let rows = Heads { outer: t, row: hd };
    let krows = Heads { outer: t, row: ds };
    let per_step = Heads { outer: t, row: 1 };
    let x = rows.input(&ctx, 31, |v| 0.5 * v);
    let q = krows.input(&ctx, 32, |v| 0.5 * v);
    let kk = krows.input(&ctx, 33, |v| 0.5 * v);
    let qk = per_step.input(&ctx, 34, |v| 0.5 * v);
    // Decay cumsums falling from zero inside the chunk, as the chunked
    // forward writes them, the same for every head.
    let mut da = Vec::with_capacity(nc * NH * CS);
    let steps = det(CS, 35, 0.05);
    for _ in 0..nc * NH {
        let mut acc = 0.0f32;
        da.extend(steps.iter().map(|v| {
            acc -= v.abs();
            acc
        }));
    }
    let da = floats(&ctx, &da);
    let states = Heads {
        outer: nc,
        row: hd * ds,
    }
    .input(&ctx, 36, |v| 0.5 * v);
    let d = Heads { outer: 1, row: 1 }.input(&ctx, 37, |v| v);
    let coop_cfg =
        chunk_scan_coop_cfg(1, nc, NH, hd, ds, CS).expect("the cooperative tiles fit this shape");
    for (what, f, cfg) in [
        ("cooperative", &k.m3_chunk_scan_fwd_coop.i32, coop_cfg),
        (
            "sliced",
            &k.m3_chunk_scan_fwd_sliced_typed.i32.f32,
            chunk_scan_sliced_cfg(1, nc, NH, hd),
        ),
    ] {
        assert!(cfg.grid_dim.1 > GRID_YZ_EXTENT, "{what}");
        let y = poisoned(&ctx, rows.len());
        let mut args: Vec<Arg> = [&y, &x, &q, &kk, &qk, &da, &states, &d]
            .iter()
            .map(|b| Arg::P(b.cached_ptr()))
            .collect();
        args.extend([1, t, NH, hd, ds, CS].map(|v| Arg::I(v as i32)));
        launch_runs_packed(&ctx, f, cfg, &args);
        assert!(
            rows.twins_match(&words(&ctx.stream, &y)),
            "chunk scan, {what} tiles"
        );
    }
}

/// The pre-state over heads past the y extent, on its fused kernel of one
/// head per block and on its sliced kernel: every output of a twin head
/// carries its first-run twin's bits, the decay cumsum among them.
#[test]
fn the_pre_state_gives_heads_past_the_y_extent_their_twins_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let k = compile(&ctx);
    let (t, hd, ds) = (CS, 4usize, 4usize);
    let nc = t.div_ceil(CS);
    let rows = Heads { outer: t, row: hd };
    let krows = Heads { outer: t, row: ds };
    let per_step = Heads { outer: t, row: 1 };
    let states = Heads {
        outer: nc,
        row: hd * ds,
    };
    let cumsums = Heads { outer: nc, row: CS };
    let x = rows.input(&ctx, 41, |v| 0.5 * v);
    let q = krows.input(&ctx, 42, |v| 0.5 * v);
    let kk = krows.input(&ctx, 43, |v| 0.5 * v);
    let dt = per_step.input(&ctx, 44, |v| 0.01 + 0.4 * v.abs());
    let trap = per_step.input(&ctx, 45, |v| 0.5 + 0.4 * v);
    let adt = per_step.input(&ctx, 46, |v| -0.05 * v.abs());
    let fused_cfg = chunk_fused_cfg(1, nc, NH, hd, ds, CS, k.smem_optin)
        .expect("the fused tile fits this shape");
    for (what, f, cfg) in [
        (
            "fused",
            &k.m3_chunk_pre_state_fused_typed.i32.f32,
            fused_cfg,
        ),
        (
            "sliced",
            &k.m3_chunk_pre_state_sliced_typed.i32.f32,
            chunk_pre_state_sliced_cfg(1, nc, NH, hd),
        ),
    ] {
        assert!(cfg.grid_dim.1 > GRID_YZ_EXTENT, "{what}");
        let layouts = [krows, per_step, per_step, per_step, states, cumsums];
        let outs = layouts.map(|h| poisoned(&ctx, h.len()));
        let mut args: Vec<Arg> = outs[..5].iter().map(|b| Arg::P(b.cached_ptr())).collect();
        args.extend([&kk, &q, &dt, &trap, &x, &adt].map(|b| Arg::P(b.cached_ptr())));
        args.push(Arg::P(outs[5].cached_ptr()));
        args.extend([1, t, NH, hd, ds, CS].map(|v| Arg::I(v as i32)));
        launch_runs_packed(&ctx, f, cfg, &args);
        let names = [
            "K_scaled",
            "qk_dot",
            "scale",
            "gamma",
            "chunk_states",
            "dA_cumsum",
        ];
        for ((layout, buf), name) in layouts.iter().zip(&outs).zip(names) {
            assert!(
                layout.twins_match(&words(&ctx.stream, buf)),
                "pre-state, {what} kernel: {name}"
            );
        }
    }
}
