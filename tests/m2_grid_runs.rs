//! Mamba-2 kernels over grid axes past the y extent of 65535 blocks. Such
//! an axis is launched in runs, each run passing its first block. A head of
//! the decode step in a later run must come out with the bits of its twin,
//! a head of the first run given the same inputs, and so must a state
//! element of the state passing, whose elements fill the y axis in blocks
//! of 256. Every other split launch of the family is proven at ordinary
//! shapes by the suites run with `MAMBA_RS_GRID_RUN`.
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use std::sync::Arc;

use cudarc::driver::sys::CUresult::CUDA_SUCCESS;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{GRID_YZ_EXTENT, y_runs, yz_runs};
use mamba_rs::mamba2::gpu::kernels::{CHUNK, Mamba2Kernels, decode_rows, step_cfg};

/// Units `u < PAIRS` and their twins `u + shift`, two full runs on.
const PAIRS: usize = 16;

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
    F(f32),
}

fn launch(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    let mut b = ctx.stream.launch_builder(f);
    for a in args {
        match a {
            Arg::P(p) => b.arg(p),
            Arg::I(i) => b.arg(i),
            Arg::F(x) => b.arg(x),
        };
    }
    unsafe { b.launch(cfg) }.expect("launch");
}

/// A layout of `outer` blocks of `units` units of `row` values, the units
/// `u < PAIRS` twinned with `u + shift`.
#[derive(Clone, Copy)]
struct Twins {
    outer: usize,
    units: usize,
    row: usize,
    shift: usize,
}

impl Twins {
    fn len(self) -> usize {
        self.outer * self.units * self.row
    }

    fn pair_up(self, v: &mut [f32]) {
        for o in 0..self.outer {
            for u in 0..PAIRS {
                let from = (o * self.units + u) * self.row;
                v.copy_within(from..from + self.row, from + self.shift * self.row);
            }
        }
    }

    fn input(self, ctx: &GpuCtx, seed: u32, scale: f32) -> GpuBuffer {
        let mut v = det(self.len(), seed, scale);
        self.pair_up(&mut v);
        floats(ctx, &v)
    }

    fn twins_match(self, w: &[u32]) -> bool {
        assert_eq!(w.len(), self.len());
        (0..self.outer).all(|o| {
            (0..PAIRS).all(|u| {
                let a = (o * self.units + u) * self.row;
                let b = a + self.shift * self.row;
                w[a..a + self.row] == w[b..b + self.row]
            })
        })
    }
}

/// The decode step over heads past the y extent: every twin head's output
/// and state carry the bits of its first-run twin. One group, so every head
/// reads the same B and C.
#[test]
fn the_decode_step_gives_heads_past_the_y_extent_their_twins_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let k = Mamba2Kernels::compile(ctx.stream.context(), arch::arch0()).expect("M2 kernels");
    let shift = 2 * 65536;
    let nh = shift + PAIRS;
    let (batch, hd, ds, ng) = (2usize, 2usize, 4usize, 1usize);
    let d_ssm = nh * hd;
    let conv_dim = d_ssm + 2 * ng * ds;
    let heads = |outer: usize, row: usize| Twins {
        outer,
        units: nh,
        row,
        shift,
    };
    // The x rows of every token lead the conv rows, which end in B and C.
    let x = heads(batch, hd);
    let mut xbc = vec![0.0f32; batch * conv_dim];
    let mut xv = det(x.len(), 1, 1.0);
    x.pair_up(&mut xv);
    let bc = det(batch * 2 * ng * ds, 2, 1.0);
    for b in 0..batch {
        let row = &mut xbc[b * conv_dim..(b + 1) * conv_dim];
        row[..d_ssm].copy_from_slice(&xv[b * d_ssm..(b + 1) * d_ssm]);
        row[d_ssm..].copy_from_slice(&bc[b * 2 * ng * ds..(b + 1) * 2 * ng * ds]);
    }
    let xbc = floats(&ctx, &xbc);
    // The projection rows hold only the step sizes here.
    let proj = heads(batch, 1).input(&ctx, 3, 1.0);
    let dt_bias = heads(1, 1).input(&ctx, 4, 0.5);
    let a_neg = {
        let t = heads(1, 1);
        let mut v: Vec<f32> = det(t.len(), 5, 1.0)
            .into_iter()
            .map(|v| -0.1 - v.abs())
            .collect();
        t.pair_up(&mut v);
        floats(&ctx, &v)
    };
    let d = heads(1, 1).input(&ctx, 6, 1.0);
    let state = heads(batch, hd * ds);
    let ssm = state.input(&ctx, 7, 0.5);
    let y = poisoned(&ctx, x.len());
    let rows = decode_rows((batch, nh), (hd, ds), k.smem_optin).expect("decode rows");
    let cfg = step_cfg(batch, nh, (hd, ds), rows);
    assert!(cfg.grid_dim.1 > GRID_YZ_EXTENT);
    let args = [
        Arg::P(y.cached_ptr()),
        Arg::P(ssm.cached_ptr()),
        Arg::P(xbc.cached_ptr()),
        Arg::P(proj.cached_ptr()),
        Arg::P(dt_bias.cached_ptr()),
        Arg::P(a_neg.cached_ptr()),
        Arg::P(d.cached_ptr()),
        Arg::I(batch as i32),
        Arg::I(nh as i32),
        Arg::I(ng as i32),
        Arg::I(hd as i32),
        Arg::I(ds as i32),
        Arg::I(conv_dim as i32),
        Arg::I(nh as i32),
        Arg::I(0),
        Arg::I(0),
        Arg::I(0),
        Arg::F(0.0),
        Arg::F(f32::INFINITY),
    ];
    for (y_first, z_first, cfg) in yz_runs(cfg) {
        let mut run = args.to_vec();
        run.extend([Arg::I(y_first), Arg::I(z_first)]);
        launch(&ctx, k.step_fwd.get(WeightDtype::F32), cfg, &run);
    }
    assert!(x.twins_match(&words(&ctx.stream, &y)), "y");
    assert!(state.twins_match(&words(&ctx.stream, &ssm)), "state");
}

/// The state passing over a head whose state elements fill more blocks of
/// 256 than the y extent holds: every twin element's entering states and
/// final state carry the bits of its first-run twin.
#[test]
fn the_state_passing_gives_elements_past_the_y_extent_their_twins_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let k = Mamba2Kernels::compile(ctx.stream.context(), arch::arch0()).expect("M2 kernels");
    let shift = 2 * GRID_YZ_EXTENT as usize * 256;
    let ds = 256usize;
    let hd = (shift + PAIRS).div_ceil(ds);
    let pn = hd * ds;
    let (batch, nh, seq) = (1usize, 1usize, 2 * CHUNK);
    let nc = seq.div_ceil(CHUNK);
    let elems = |outer: usize| Twins {
        outer,
        units: pn,
        row: 1,
        shift,
    };
    let states = elems(nc).input(&ctx, 11, 0.5);
    let ssm = elems(1).input(&ctx, 12, 0.5);
    let acs: Vec<f32> = det(batch * nc * nh * CHUNK, 13, 0.05)
        .into_iter()
        .map(|v| -v.abs())
        .collect();
    let acs = floats(&ctx, &acs);
    let cfg = LaunchConfig {
        grid_dim: ((batch * nh) as u32, pn.div_ceil(256) as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    assert!(cfg.grid_dim.1 > GRID_YZ_EXTENT);
    let args = [
        Arg::P(states.cached_ptr()),
        Arg::P(ssm.cached_ptr()),
        Arg::P(acs.cached_ptr()),
        Arg::I(batch as i32),
        Arg::I(seq as i32),
        Arg::I(nh as i32),
        Arg::I(hd as i32),
        Arg::I(ds as i32),
    ];
    for (y_first, cfg) in y_runs(cfg) {
        let mut run = args.to_vec();
        run.push(Arg::I(y_first));
        launch(&ctx, &k.state_passing, cfg, &run);
    }
    assert!(
        elems(nc).twins_match(&words(&ctx.stream, &states)),
        "entering states"
    );
    assert!(
        elems(1).twins_match(&words(&ctx.stream, &ssm)),
        "final state"
    );
}
